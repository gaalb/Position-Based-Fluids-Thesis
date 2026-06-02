// Root signature:
//   CBV(b0)                                    -- ComputeCb (for boxMin, boxMax, h -> gridDims)
//   DescriptorTable(UAV(u0, numDescriptors=5)) -- u1: cellCount (read), u4: cellPrefixSum (write)

#define PrefixSumRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 5))"

#include "GridUtils.hlsli" // gridDims()

ByteAddressBuffer cellCount : register(u1);
RWByteAddressBuffer cellPrefixSum : register(u4);
RWByteAddressBuffer perPageSum : register(u2);

#define rowSize 32
#define nRowsPerPage 32
#define groupDivisor 4

groupshared uint s[nRowsPerPage];

[RootSignature(PrefixSumRootSig)]
[numthreads(rowSize, nRowsPerPage / groupDivisor, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint localasses[groupDivisor];
    for (int did = 0; did < groupDivisor; did++)
    {
        uint rid = tid.y + did * nRowsPerPage / groupDivisor;
        uint flatid = (rid << 5) | tid.x;
        uint elementIndex = flatid + gid.x * rowSize * nRowsPerPage;
        uint count = cellCount.Load(elementIndex << 2);
        localasses[did] = WaveActiveSum(count);
        if (tid.x == 31)
        {
            s[rid] = localasses[did] + count;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.y == 0)
    {
        uint perRowSum =  WaveActiveSum(s[tid.x]);
        if (tid.x == 31) {
            perPageSum.Store(gid.x << 2, perRowSum + s[31]);
        }
        s[tid.x] = perRowSum;
    }
    GroupMemoryBarrierWithGroupSync();
    for (int did = 0; did < groupDivisor; did++)
    {
        uint rid = tid.y + did * nRowsPerPage / groupDivisor;
        uint flatid = (rid << 5) | tid.x;
        uint elementIndex = flatid + gid.x * rowSize * nRowsPerPage;
        cellPrefixSum.Store(elementIndex << 2, localasses[did] + s[rid]);
    }
}