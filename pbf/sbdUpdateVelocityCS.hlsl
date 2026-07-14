// Soft body dynamics: derive new velocity from the position change and commit the predicted position.
// Inputs : position (u0), predictedPosition (u2)
// Outputs: velocity (u1) updated as (predPos - pos) / dt; position (u0) set to predictedPosition
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
    // TODO: update velocity from (predictedPosition - position) / dt and commit position
    velocity[id.x].xyz = (predictedPosition[id.x].xyz - position[id.x].xyz) / dt * 0.9;
    position[id.x].xyz = predictedPosition[id.x].xyz;
}
