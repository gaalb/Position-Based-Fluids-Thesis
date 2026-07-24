// Soft body dynamics: commit the predicted position and derive the new velocity.
// 1. Hard clamp pred to [boxMin, boxMax] as a backstop against tunneling through the soft exponential wall.
// 2. Derive velocity: v = (pred - pos) / dt  (implicit position update rule).
// 3. Commit: position = pred.
// Inputs : position (u0), predictedPosition (u2)
// Outputs: velocity (u1), position (u0)
#define SbdUpdateVelocityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=3))"

RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

#include "ComputeCb.hlsli"

[RootSignature(SbdUpdateVelocityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SBD_NUM_NODES) return;

    float3 pred = clamp(predictedPosition[id.x], boxMin, boxMax);
    velocity[id.x] = (pred - position[id.x]) / dt;
    position[id.x] = pred;
}
