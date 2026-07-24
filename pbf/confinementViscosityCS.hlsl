// Combines vorticity confinement, XSPH viscosity, and SBD drag into a single neighbor pass.
//
// Confinement steps per particle i:
//   1. eta_i = sum_{j != i} |omega_j| * grad_W_spiky(r_ij, h)
//   2. N_i   = eta_i / |eta_i|
//   3. f_i   = vorticityEpsilon * (N_i x omega_i)
//   4. v_i  += dt * f_i
//
// Viscosity (XSPH) per particle i:
//   v_new_i = v_i + c * sum_{j != i} (v_j - v_i) * W_poly6(r_ij, h)
//
// SBD drag per particle i (applied using original v_i, before confinement):
//   v_new_i += sbdDragStrength * sum_{j in SBD neighbors} (v_sbd_j - v_i) * smoothstep(r/SBD_H)
//
// In: position, velocity, omega, cellCount, cellPrefixSum, sbdPosition, sbdVelocity, sbdCellCount, sbdCellPrefixSum, sbdNodeList
// Out: scratch (new velocity, Jacobi mode)

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

[RootSignature(ConfinementViscosityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = position[i];
    float3 vi = velocity[i];
    float3 omegaI = omega[i];

    float3 eta = float3(0, 0, 0); // accumulates grad(|omega|) for confinement
    float3 xsphSum = float3(0, 0, 0); // accumulates weighted velocity differences for viscosity

    // The grid was built from predictedPositions, but this pass uses old positions.
    // See vorticityCS for the justification of this approximation.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r  = pi - position[j];
            float  r2 = dot(r, r);

            // Confinement: accumulate eta_i = sum_j |omega_j| * grad_W_spiky(r_ij)
            eta += length(omega[j]) * SpikyGrad(r, r2);

            // Viscosity: accumulate XSPH sum = sum_j (v_j - v_i) * W_poly6(r_ij)
             // (v_j - v_i) * W: neighbor's velocity contribution weighted by proximity.
            // The full XSPH formula (Schechter & Bridson 2012) is:
            //   sum_j (m_j / rho_j) * (v_j - v_i) * W(r_ij, h)
            // m_j = 1 is dropped as a uniform constant; it only scales the sum and is
            // absorbed into the viscosity coefficient c.
            // 1/rho_j is also dropped. Unlike m_j, rho_j varies per particle, so omitting it
            // changes the relative weighting of neighbors and is not trivially justified.
            // The assumption is that PBF keeps rho_j ~ rho0 for all j (incompressibility),
            // making 1/rho_j approximately uniform. Under that assumption it too is absorbed
            // into c, and the formula reduces to what we compute here.
            xsphSum += (velocity[j] - vi) * Poly6(r, r2);
        }
    }

    // SBD drag: pull fluid velocity toward nearby SBD node velocities (uses original vi)
    float3 sbdDrag = float3(0, 0, 0);
    SbdNeighborCells sbdNhbrs = SbdNeighborCellIndices(pi);
    for (uint dc = 0; dc < sbdNhbrs.count; dc++) {
        uint dci = sbdNhbrs.indices[dc];
        uint dcount = sbdCellCount[dci];
        for (uint dn = 0; dn < dcount; dn++)
        {
            uint j = sbdNodeList[sbdCellPrefixSum[dci] + dn];
            float3 r = pi - sbdPosition[j];
            float dist2 = dot(r, r);
            if (dist2 < SBD_H * SBD_H)
            {
                float dist = sqrt(dist2);
                float t = 1.0f - dist / SBD_H;
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

    scratch[i] = vi + viscosity * xsphSum + sbdDragStrength * 10.0 * sbdDrag;
}
