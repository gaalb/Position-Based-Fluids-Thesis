// Velocity matching: pull fluid velocity toward nearby character particle velocities.
// Run after velocityFromScratchShader (XSPH done) and before updatePositionShader.
// This makes the fluid visibly "follow" the character's animated motion.
//
// For each nearby character particle j within SBD_H:
//   drag_i += (charVelocity_j - v_i) * smoothstep(r/SBD_H)
//
// In:  fluidVelocity (u0, back — modified in-place), fluidPosition (u1, back — for grid lookup),
//      charPosition (u2), charVelocity (u3),
//      charCellCount (u4), charCellPrefixSum (u5), charNodeList (u6)
// Out: fluidVelocity (u0) corrected in-place
#define CharVelInfluenceRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=7))"

RWStructuredBuffer<float3> fluidVelocity     : register(u0);
RWStructuredBuffer<float3> fluidPosition     : register(u1);
RWStructuredBuffer<float3> charPosition      : register(u2);
RWStructuredBuffer<float3> charVelocity      : register(u3);
RWStructuredBuffer<uint>   charCellCount     : register(u4);
RWStructuredBuffer<uint>   charCellPrefixSum : register(u5);
RWStructuredBuffer<uint>   charNodeList      : register(u6);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(CharVelInfluenceRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= numParticles) return;

    float3 pos = fluidPosition[id.x];
    float3 v   = fluidVelocity[id.x];
    float3 drag = float3(0, 0, 0);

    SbdNeighborCells nCells = SbdNeighborCellIndices(pos);
    for (uint c = 0; c < nCells.count; c++) {
        uint ci = nCells.indices[c];
        uint count = charCellCount[ci];
        for (uint s = 0; s < count; s++) {
            uint j = charNodeList[charCellPrefixSum[ci] + s];
            float3 r = pos - charPosition[j];
            float dist2 = dot(r, r);
            if (dist2 < SBD_H * SBD_H) {
                float dist = sqrt(dist2);
                float t = 1.0f - dist / SBD_H;
                float w = 3.0f * t * t - 2.0f * t * t * t; // smoothstep
                drag += (charVelocity[j] - v) * w;
            }
        }
    }
    fluidVelocity[id.x] = v + charVelocityStrength * drag;
}
