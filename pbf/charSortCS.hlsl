// Scatter character particle indices into charNodeList using prefix-summed cell offsets.
// Run after the prefix-sum passes and a second clear of charCellCount.
// charNodeList[charCellPrefixSum[ci] + slot] = particle_index
// In:  charPosition (u0), charCellCount (u1, pre-cleared), charCellPrefixSum (u2)
// Out: charNodeList (u3), charCellCount (u1, restored by atomic side-effect)
#define CharSortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=4))"

RWStructuredBuffer<float3> charPosition      : register(u0);
RWStructuredBuffer<uint>   charCellCount     : register(u1);
RWStructuredBuffer<uint>   charCellPrefixSum : register(u2);
RWStructuredBuffer<uint>   charNodeList      : register(u3);

#include "ComputeCb.hlsli"
#include "SbdGridUtils.hlsli"

[RootSignature(CharSortRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= N_CHAR_PARTICLES) return;
    int3 cell = sbdPosToCell(charPosition[id.x]);
    uint ci = sbdCellIndex(cell);
    uint slot;
    InterlockedAdd(charCellCount[ci], 1, slot);
    charNodeList[charCellPrefixSum[ci] + slot] = id.x;
}
