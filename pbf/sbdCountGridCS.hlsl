// Count how many SBD nodes fall into each grid cell.
// First pass of the SBD spatial-grid build; clears must run before this.
// In:  sbdPosition (u0) — committed SBD positions from the previous frame
//      sbdCellCount (u1) — pre-cleared to zero
// Out: sbdCellCount (u1) — atomically incremented per cell
#define SbdCountGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=2))"

RWStructuredBuffer<float3> sbdPosition  : register(u0);
RWStructuredBuffer<uint>   sbdCellCount : register(u1);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(SbdCountGridRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SBD_NUM_NODES ||
        id.x == SBD_NUM_CUBIC_NODES+0 ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1) - 1 ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1) - 1 ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1) - (SBD_DIM_X+1) ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1)*(SBD_DIM_Z+1) - 1 ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1)*(SBD_DIM_Z+1) - (SBD_DIM_X+1) ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1)*(SBD_DIM_Z+1) - 1-(SBD_DIM_X+1)*SBD_DIM_Y ||
        id.x == SBD_NUM_CUBIC_NODES+(SBD_DIM_X+1)*(SBD_DIM_Y+1)*(SBD_DIM_Z+1) - (SBD_DIM_X+1)-(SBD_DIM_X+1)*SBD_DIM_Y 
    ) return;

    int3 cell = sbdPosToCell(sbdPosition[id.x]);
    InterlockedAdd(sbdCellCount[sbdCellIndex(cell)], 1);
}
