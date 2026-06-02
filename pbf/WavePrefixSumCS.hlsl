#define SortSig "RootFlags( 0 )," \
				"RootConstants(num32BitConstants=1, b0)," \
                "DescriptorTable(UAV(u0, numDescriptors=1), UAV(u1, numDescriptors=1), UAV(u2, numDescriptors=1), UAV(u3, numDescriptors=1), UAV(u4, numDescriptors=1))" 

// uav offset @mortons or @ (#0 or #4)
RWByteAddressBuffer dc_keys : register(u0);
RWByteAddressBuffer perPageBucketOffsets : register(u1);
RWByteAddressBuffer dc_indicesWithKeyBits0 : register(u2);
RWByteAddressBuffer dc_indicesWithKeyBits1 : register(u3);
RWByteAddressBuffer globalBucketOffsets : register(u4);
uint maskOffsets : register(b0);

#define groupSize 32
#define nBuckets 16
#define nPagesPerChunk 32
#define nChunks 32
#define rowSize 32
#define nRowsPerPage 32

groupshared uint chunkTotals[nChunks];

[RootSignature(SortSig)]
[numthreads(32, 32, 1)]
void csScan(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint iPageInChunk = tid.x;
    uint iChunk = tid.y;
    uint iBucket = gid.x;
    uint pGlobal = (iChunk * nPagesPerChunk + iPageInChunk + iBucket * nPagesPerChunk * nChunks) << 2;

    uint cL = perPageBucketOffsets.Load(pGlobal);
    uint inChunkSum = WavePrefixSum(cL) + cL;
    if (iPageInChunk == 31)
    {
        chunkTotals[iChunk] = inChunkSum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (iChunk == 0)
    {
        chunkTotals[tid.x] = WavePrefixSum(chunkTotals[tid.x]);
    }
    GroupMemoryBarrierWithGroupSync();
    uint cLL = chunkTotals[iChunk] + inChunkSum;
    
    globalBucketOffsets.Store(pGlobal, cLL);
}