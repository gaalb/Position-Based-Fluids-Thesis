// Assign each SBD node a slot in its grid cell and record its index in sbdNodeList.
// Run after the prefix-sum passes and a second clear of sbdCellCount.
// sbdNodeList[sbdCellPrefixSum[ci] + slot] = node_index
// After this pass sbdCellCount[ci] equals the node count for cell ci again (side-effect of atomic).
// In:  sbdPosition (u0), sbdCellCount (u1, pre-cleared), sbdCellPrefixSum (u2)
// Out: sbdNodeList (u3), sbdCellCount (u1, restored by atomic side-effect)
#define SbdSortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=4))"

RWStructuredBuffer<float3> sbdPosition      : register(u0);
RWStructuredBuffer<uint>   sbdCellCount     : register(u1);
RWStructuredBuffer<uint>   sbdCellPrefixSum : register(u2);
RWStructuredBuffer<uint>   sbdNodeList      : register(u3);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"
#include "SharedConfig.hlsli"

[RootSignature(SbdSortRootSig)]
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
    uint ci = sbdCellIndex(cell);
    uint slot;
    InterlockedAdd(sbdCellCount[ci], 1, slot);
    sbdNodeList[sbdCellPrefixSum[ci] + slot] = id.x;
}
