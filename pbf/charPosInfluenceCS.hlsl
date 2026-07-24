// Suction: pull fluid predicted positions toward character particle positions.
// Run after the PBF solver loop, before updateVelocityCS (so the velocity update
// naturally picks up the displacement as a velocity change).
//
// For each nearby character particle j within SBD_H:
//   pull_i += (charPos_j - predPos_i) * smoothstep(r/SBD_H) * densityDeficit_j
//
// densityDeficit_j = max(0, 1 - charDensity_j / RHO0)
// This scales the suction down when the character volume is already full of fluid.
//
// In:  fluidPredPos (u0, back), charPosition (u1), charDensity (u2),
//      charCellCount (u3), charCellPrefixSum (u4), charNodeList (u5)
// Out: fluidPredPos (u0) corrected in-place
#define CharPosInfluenceRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=6))"

RWStructuredBuffer<float3> fluidPredPos      : register(u0);
RWStructuredBuffer<float3> charPosition      : register(u1);
RWStructuredBuffer<float>  charDensity       : register(u2);
RWStructuredBuffer<uint>   charCellCount     : register(u3);
RWStructuredBuffer<uint>   charCellPrefixSum : register(u4);
RWStructuredBuffer<uint>   charNodeList      : register(u5);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(CharPosInfluenceRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= numParticles) return;

    float3 pos = fluidPredPos[id.x];
    float3 pull = float3(0, 0, 0);

    SbdNeighborCells nCells = SbdNeighborCellIndices(pos);
    for (uint c = 0; c < nCells.count; c++) {
        uint ci = nCells.indices[c];
        uint count = charCellCount[ci];
        for (uint s = 0; s < count; s++) {
            uint j = charNodeList[charCellPrefixSum[ci] + s];
            float3 r = charPosition[j] - pos; // points toward char particle
            float dist2 = dot(r, r);
            if (dist2 > 1e-8f && dist2 < SBD_H * SBD_H) {
                float dist = sqrt(dist2);
                float t = 1.0f - dist / SBD_H;
                float w = 3.0f * t * t - 2.0f * t * t * t; // smoothstep
                //float deficit = max(0.0f, 1.0f - charDensity[j] / RHO0);
                pull += r * (w / dist) ;// * deficit;
            }
        }
    }
    fluidPredPos[id.x] = pos + pull * charSuctionStrength;
}
