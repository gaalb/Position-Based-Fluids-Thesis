// Soft body dynamics: predict positions by integrating velocity forward one timestep.
// Inputs : position (u0), velocity (u1)
// Output : predictedPosition (u2)
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
    // TODO: integrate gravity + external forces into velocity, then predict position
	predictedPosition[id.x].xyz = position[id.x].xyz + velocity[id.x].xyz * dt;
}
