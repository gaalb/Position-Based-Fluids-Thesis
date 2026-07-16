// Soft body dynamics: predict positions by integrating velocity forward one timestep.
// 1. Apply gravity.
// 2. Velocity-level box wall collision (clamp inward velocity component, damp tangential).
// 3. Predict position: p* = pos + v * dt.
// Inputs : position (u0), velocity (u1)
// Output : predictedPosition (u2), velocity (u1) updated
#define SbdPredictRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=3))"

RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

#include "ComputeCb.hlsli"

[RootSignature(SbdPredictRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SBD_NUM_NODES) return;

    float3 pos = position[id.x];
    float3 v   = velocity[id.x];

   // v += float3(0.0, -9.8, 0.0) * dt;

    // velocity-level box wall collision
    if (pos.x <= boxMin.x) { v.x = max(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.x >= boxMax.x) { v.x = min(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y <= boxMin.y) { v.y = max(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y >= boxMax.y) { v.y = min(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.z <= boxMin.z) { v.z = max(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }
    if (pos.z >= boxMax.z) { v.z = min(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }

    velocity[id.x]          = v;
    predictedPosition[id.x] = pos + v * dt;
}
