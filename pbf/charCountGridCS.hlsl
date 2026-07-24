// Count how many character particles fall into each SBD-grid cell.
// Uses the same SBD_H cell size / mapping as SbdGridUtils so charInfluenceCS can do
// 27-cell neighbor searches identical to the SBD drag lookups.
// In:  charPosition (u0) — current world-space skinned positions
//      charCellCount (u1) — pre-cleared to zero
// Out: charCellCount (u1) — atomically incremented per cell
#define CharCountGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=2))"

RWStructuredBuffer<float3> charPosition  : register(u0);
RWStructuredBuffer<uint>   charCellCount : register(u1);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(CharCountGridRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= N_CHAR_PARTICLES) return;
    int3 cell = sbdPosToCell(charPosition[id.x]);
    InterlockedAdd(charCellCount[sbdCellIndex(cell)], 1);
}
