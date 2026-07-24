// Pore suction: each fluid particle is attracted toward nearby SBD nodes.
// Uses the SBD spatial grid built from committed SBD positions this frame.
// Applies a direct position correction to fluidPredPos (the back buffer, sorted by the PBF grid).
// Run after the PBF constraint-solver loop, before updateVelocityCS.
//
// The correction vector for one SBD node j at distance r < H:
//   pull += (r_hat) * (1 - r/H)^2
// Total correction applied: fluidPredPos += pull * sbdSuctionStrength
//
// In:  fluidPredPos (u0, back buffer), sbdPosition (u1)
//      sbdCellCount (u2), sbdCellPrefixSum (u3), sbdNodeList (u4)
// Out: fluidPredPos (u0, corrected in-place)
#define SbdPoreSuctionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=5))"

RWStructuredBuffer<float3> fluidPredPos     : register(u0);
RWStructuredBuffer<float3> sbdPosition      : register(u1);
RWStructuredBuffer<uint>   sbdCellCount     : register(u2);
RWStructuredBuffer<uint>   sbdCellPrefixSum : register(u3);
RWStructuredBuffer<uint>   sbdNodeList      : register(u4);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(SbdPoreSuctionRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= numParticles) return;

    float3 pos = fluidPredPos[id.x];
    float3 pull = float3(0, 0, 0);

    SbdNeighborCells nCells = SbdNeighborCellIndices(pos);
    for (uint c = 0; c < nCells.count; c++) {
        uint ci = nCells.indices[c];
        uint nodeCount = sbdCellCount[ci];
        uint base = sbdCellPrefixSum[ci];
        for (uint s = 0; s < nodeCount; s++) {
            uint j = sbdNodeList[base + s];
            float3 r = sbdPosition[j] - pos;
            float dist2 = dot(r, r);
            if (dist2 > 1e-8f && dist2 < SBD_H * SBD_H) {
                float dist = sqrt(dist2);
                //float t = 1.0f - dist2 / (SBD_H * SBD_H); // 1 - (r/H)²
                //float w = t * t * t;                        // Wyvill: (1 - (r/H)²)³
                float t = 1.0f - dist / SBD_H;
                float w = 3.0 * t * t - 2.0 * t * t * t; // Smoothstep: 3t² - 2t³ 
                pull += r * (w / dist);
            }
        }
    }

    fluidPredPos[id.x] = pos + pull * sbdSuctionStrength;
}
