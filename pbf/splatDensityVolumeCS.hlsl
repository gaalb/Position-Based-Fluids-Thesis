// splatDensityVolumeCS.hlsl
// Per-particle pass of the splat density pipeline.
//
// Algorithm:
//   For each particle, compute the bounding box of voxels whose centers lie within
//   the SPH kernel radius H, then evaluate Poly6 for each candidate voxel and
//   atomically accumulate the contribution (as a float stored in raw bits) into densityVolume
//   via a compare-and-swap loop. After the draw, DrawLiquidSurface clears densityVolume to 0
//   via ClearUnorderedAccessViewUint, ready for the next frame.
//
// Dispatched from the GRAPHICS queue (DIRECT command list) in DrawLiquidSurface().
// In:  position
// Out: densityVolume     

#define SplatDensityVolumeRootSig "CBV(b0), DescriptorTable(SRV(t0, numDescriptors = 1), UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

StructuredBuffer<float3> position   : register(t0);
RWTexture3D<uint>        densityVolume : register(u0); // raw float bits; asfloat(densityVolume[v]) gives density

// Atomically adds 'value' to densityVolume[coord] using a compare-and-swap loop.
// densityVolume stores raw IEEE 754 float bits as uint. asfloat/asuint are pure bit
// reinterpretations (no arithmetic conversion), so the float addition is exact.
// The loop retries if another thread modified the slot between our read and swap.
// Poly6 contributions are always positive, so asfloat is always >= 0: no sign-bit issues.
void FloatInterlockedAdd(uint3 coord, float value)
{
    uint assumed, old;
    old = densityVolume[coord]; // initial read; may be modified by another thread before we write back
    [loop] do {
        assumed = old;
        // asfloat(assumed) is a float, add value to it to get the new desired float value
        // bit-cast it into uint, we get asuint(asfloat(assumed) + value), the value we want to store 
        //into densityVolume[coord] if no other thread has modified it since our last read.
        // InterlockedCompareExchange reads the current densityVolume[coord] into old (output parameter),
        // compares if densityVolume[coord] is still equal to assumed, and if so, writes the new value.
        InterlockedCompareExchange(densityVolume[coord], assumed, asuint(asfloat(assumed) + value), old);
    // If old is not equal to assumed, another thread wrote to densityVolume[coord] between our previous
    // read and this CAS attempt. The value we computed (asfloat(assumed) + value) was based of a stale 
    // assumed, so we must retry. On the next iteration, old now holds the actual value, since it gets
    // set by InterlockedCompareExchange whether the swap succeeded or not. Store it into assumed, 
    // recompute the addition from the correct base and try again. 
    } while (old != assumed);
}

[RootSignature(SplatDensityVolumeRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint pIdx = id.x;
    if (pIdx >= numParticles)
        return;

    float3 pos = position[pIdx];

    // pos-GRID_MIN is the float world coordinate position of this particle, 
    // dividing by VOXEL_SIZE gets us that in voxel steps. Voxel v has its center
    // at world position GRID_MIN + (v + 0.5) * voxelSize.
    float3 voxelF = (pos - GRID_MIN) / VOXEL_SIZE;

    // Conservative bounding box in integer voxel indices.
    // voxelF - H_IN_VOXEL is the continuous voxel coordinate of the point that
    // is exactly one SPH radius away in the negative direction along each axis,
    // the floor of that gives its integer voxel coordinate. We clamp it to 0 to
    // avoid negative indices. Same idea in the positive direction for vMax.
    int3 vMin = max(int3(0, 0, 0), int3(floor(voxelF - H_IN_VOXEL)));
    int3 vMax = min(int3(VOL_DIM - 1, VOL_DIM - 1, VOL_DIM - 1), int3(floor(voxelF  + H_IN_VOXEL)));

    for (int vz = vMin.z; vz <= vMax.z; vz++)
    for (int vy = vMin.y; vy <= vMax.y; vy++)
    for (int vx = vMin.x; vx <= vMax.x; vx++)
    {
        float3 voxelCenter = GRID_MIN + (float3(vx, vy, vz) + 0.5f) * VOXEL_SIZE;
        float3 r  = voxelCenter - pos;
        float  r2 = dot(r, r);

        float w = Poly6(r, r2); // 0.0 if r2 > H*H
        if (w <= 0.0f)
            continue;

        if(pos.x < 0.0) w *= 1;
        
        FloatInterlockedAdd(uint3(vx, vy, vz), w);
    }
}
