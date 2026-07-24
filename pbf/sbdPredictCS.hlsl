// Soft body dynamics: predict positions by integrating velocity forward one timestep.
// 1. Apply gravity.
// 2. Exponential wall repulsion: f = sbdWallStrength * exp(-d / sbdWallFalloff), d = signed dist inside box.
// 3. Fluid drag: v += sbdDragStrength * sum_{j in fluid neighbors} (v_fluid_j - v) * smoothstep(r/H) * dt
// 4. Predict position: p* = pos + v * dt.
// Inputs : position (u0), velocity (u1), fluidPosition (u3), fluidVelocity (u4), fluidCellCount (u5), fluidCellPrefixSum (u6)
// Output : predictedPosition (u2), velocity (u1) updated
#define SbdPredictRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=7))"

RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float3> fluidPosition     : register(u3);
RWStructuredBuffer<float3> fluidVelocity     : register(u4);
RWStructuredBuffer<uint>   fluidCellCount    : register(u5);
RWStructuredBuffer<uint>   fluidCellPrefixSum: register(u6);

#include "ComputeCb.hlsli"
#include "GridUtils.hlsli"

[RootSignature(SbdPredictRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SBD_NUM_NODES) return;

    float3 pos = position[id.x];
    float3 v   = velocity[id.x] * exp(-0.25 * dt);

    v += float3(0.0, -9.8, 0.0) * dt;

    // Soft exponential wall repulsion: each face pushes inward with force = sbdWallStrength * exp(-d / sbdWallFalloff)
    // where d is signed distance from that face (positive = inside box, negative = penetrating).
    float s = sbdWallStrength;
    float L = max(sbdWallFalloff, 1e-4f);
    float3 wallForce;
    wallForce.x = s * (exp(-(pos.x - boxMin.x) / L) - exp(-(boxMax.x - pos.x) / L));
    wallForce.y = s * (exp(-(pos.y - boxMin.y) / L) - exp(-(boxMax.y - pos.y) / L));
    wallForce.z = s * (exp(-(pos.z - boxMin.z) / L) - exp(-(boxMax.z - pos.z) / L));
    v += wallForce * dt;

    // Fluid drag: pull SBD node velocity toward local fluid velocity
    float3 fluidDrag = float3(0, 0, 0);
    NeighborCells fCells = NeighborCellIndices(pos);
    for (uint fc = 0; fc < fCells.count; fc++) {
        uint ci = fCells.indices[fc];
        uint count = fluidCellCount[ci];
        for (uint fn = 0; fn < count; fn++) {
            uint j = fluidCellPrefixSum[ci] + fn;
            float3 r = pos - fluidPosition[j];
            float dist2 = dot(r, r);
            if (dist2 < H * H) {
                float dist = sqrt(dist2);
                float t = 1.0f - dist / SBD_H;
                float w = 3.0f * t * t - 2.0f * t * t * t; // smoothstep
                fluidDrag += (fluidVelocity[j] - v) * w;
            }
        }
    }
    v += sbdDragStrength * 30.0 * fluidDrag * dt;

    velocity[id.x]          = v;
    predictedPosition[id.x] = pos + v * dt;
}
