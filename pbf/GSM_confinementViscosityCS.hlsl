// Two-phase broadcast-GSM variant of confinementViscosityCS.
//
// WHY THE PREVIOUS GSM VARIANT FAILED:
//   Each thread read gs_X[j - groupStart] where j varied per thread.
//   Different threads in the same warp hit different GSM banks simultaneously
//   -> bank conflicts -> MIOT stalls -> slower than the no-GSM baseline.
//
// THIS VARIANT: all threads step through GSM in lockstep, reading the same
//   index k at each iteration. The hardware serves gs_X[k] as a broadcast
//   (one read, fanned to all threads): zero bank conflicts, near-zero latency.
//
// Phase 1 — GSM broadcast sweep:
//   Iterate k = 0 .. THREAD_GROUP_SIZE-1. All threads read the same gs_X[k]
//   each step -> broadcast. Particles outside H contribute 0 (kernels guard
//   with r2 > H*H), so false positives (intra-group non-neighbors) are
//   correctness-safe and cost only a few cheap ALU ops. Self (k == localIdx,
//   which means groupStart + k == i) is skipped explicitly.
//
// Phase 2 — residual VRAM pass:
//   Same grid traversal as the baseline. Any j already covered by Phase 1
//   (j in [groupStart, groupStart + THREAD_GROUP_SIZE)) is skipped, since i
//   itself also falls in that range, this subsumes the j == i self-check too.
//
// GSM budget: 3 arrays * THREAD_GROUP_SIZE * 12 bytes = 9 KB < 32 KB minimum.
//
// In: position, velocity, omega, cellCount, cellPrefixSum
// Out: scratch (new velocity, Jacobi mode) or velocity (Gauss-Seidel mode)

#define ConfinementViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 11))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"
#include "SbdGridUtils.hlsli"

RWStructuredBuffer<float3> position         : register(u0);
RWStructuredBuffer<float3> velocity         : register(u1);
RWStructuredBuffer<float3> omega            : register(u2);
RWStructuredBuffer<float3> scratch          : register(u3);
RWStructuredBuffer<uint>   cellCount        : register(u4);
RWStructuredBuffer<uint>   cellPrefixSum    : register(u5);
RWStructuredBuffer<float3> sbdPosition      : register(u6);
RWStructuredBuffer<float3> sbdVelocity      : register(u7);
RWStructuredBuffer<uint>   sbdCellCount     : register(u8);
RWStructuredBuffer<uint>   sbdCellPrefixSum : register(u9);
RWStructuredBuffer<uint>   sbdNodeList      : register(u10);

groupshared float3 gs_position[THREAD_GROUP_SIZE];
groupshared float3 gs_velocity[THREAD_GROUP_SIZE];
groupshared float3 gs_omega[THREAD_GROUP_SIZE];

[RootSignature(ConfinementViscosityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchID : SV_DispatchThreadID,
    uint  localIdx   : SV_GroupIndex,
    uint3 groupID    : SV_GroupID)
{
    uint i          = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE;

    // Preload into GSM. Out-of-bounds threads write a sentinel position far
    // outside the grid so that Phase 1 gets r2 >> H*H for those slots and
    // the kernels return 0 harmlessly. Velocity and omega can stay zero.
    gs_position[localIdx] = (i < numParticles) ? position[i] : float3(1e10f, 1e10f, 1e10f);
    gs_velocity[localIdx] = (i < numParticles) ? velocity[i] : (float3)0;
    gs_omega[localIdx]    = (i < numParticles) ? omega[i]    : (float3)0;
    GroupMemoryBarrierWithGroupSync();

    if (i >= numParticles)
        return;

    float3 pi     = gs_position[localIdx];
    float3 vi     = gs_velocity[localIdx];
    float3 omegaI = gs_omega[localIdx];

    float3 eta     = float3(0, 0, 0);
    float3 xsphSum = float3(0, 0, 0);

    // Phase 1: broadcast sweep over GSM
    // k is the same for all threads at each step -> hardware broadcasts gs_X[k]
    // to all threads at once, eliminating bank conflicts entirely.
    for (uint k = 0; k < THREAD_GROUP_SIZE; k++)
    {
        if (k == localIdx)  // skip self (groupStart + k == i)
            continue;

        float3 pj = gs_position[k]; // broadcast read
        float3 r  = pi - pj;
        float  r2 = dot(r, r);

        eta     += length(gs_omega[k]) * SpikyGrad(r, r2); // broadcast read
        xsphSum += (gs_velocity[k] - vi) * Poly6(r, r2);   // broadcast read
    }

    // Phase 2: residual pass for particles outside this thread group 
    // Skip j in [groupStart, groupStart + THREAD_GROUP_SIZE): already covered
    // by Phase 1. Because i itself is in that range, this also subsumes j == i.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;

            if (j >= groupStart && j < groupStart + THREAD_GROUP_SIZE)
                continue; // handled in Phase 1

            float3 r  = pi - position[j];
            float  r2 = dot(r, r);

            eta     += length(omega[j]) * SpikyGrad(r, r2);
            xsphSum += (velocity[j] - vi) * Poly6(r, r2);
        }
    }

    // SBD drag: pull fluid velocity toward nearby SBD node velocities (uses original vi)
    float3 sbdDrag = float3(0, 0, 0);
    SbdNeighborCells sbdNhbrs = SbdNeighborCellIndices(pi);
    for (uint dc = 0; dc < sbdNhbrs.count; dc++) {
        uint dci = sbdNhbrs.indices[dc];
        uint dcount = sbdCellCount[dci];
        for (uint dn = 0; dn < dcount; dn++) {
            uint j = sbdNodeList[sbdCellPrefixSum[dci] + dn];
            float3 r = pi - sbdPosition[j];
            float dist2 = dot(r, r);
            if (dist2 < SBD_H * SBD_H) {
                float t = 1.0f - dist2 / (SBD_H * SBD_H);
                float w = 3.0f * t * t - 2.0f * t * t * t; // smoothstep
                sbdDrag += (sbdVelocity[j] - vi) * w;
            }
        }
    }

    // Confinement
    float etaLen = length(eta);
    if (etaLen >= 1e-6)
    {
        float3 N = eta / etaLen;
        vi += dt * vorticityEpsilon * cross(N, omegaI);
    }

    scratch[i] = vi + viscosity * xsphSum + sbdDragStrength * sbdDrag;
}
