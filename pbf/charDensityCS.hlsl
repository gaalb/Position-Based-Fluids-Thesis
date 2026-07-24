// Evaluate fluid SPH density at each character particle position.
// Uses the PBF spatial grid (cell size H, built from sorted positions this frame).
// The density estimate tells charPosInfluenceCS how full the character volume already is,
// so suction strength scales with deficit (1 - density/rho0).
//
// In:  charPosition (u0), fluidPosition (u2), fluidCellCount (u3), fluidCellPrefixSum (u4)
// Out: charDensity  (u1)
#define CharDensityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=5))"

RWStructuredBuffer<float3> charPosition       : register(u0);
RWStructuredBuffer<float>  charDensity        : register(u1);
RWStructuredBuffer<float3> fluidPosition      : register(u2);
RWStructuredBuffer<uint>   fluidCellCount     : register(u3);
RWStructuredBuffer<uint>   fluidCellPrefixSum : register(u4);

#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

[RootSignature(CharDensityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= N_CHAR_PARTICLES) return;

    float3 pos = charPosition[id.x];
    float density = 0.0f;

    NeighborCells nCells = NeighborCellIndices(pos);
    for (uint c = 0; c < nCells.count; c++) {
        uint ci = nCells.indices[c];
        uint count = fluidCellCount[ci];
        for (uint s = 0; s < count; s++) {
            uint j = fluidCellPrefixSum[ci] + s;
            float3 r = pos - fluidPosition[j];
            float r2 = dot(r, r);
            density += Poly6(r, r2);
        }
    }
    charDensity[id.x] = density;
}
