// Root signature shared by VS, GS, and PS
// CBV(b0): per-frame constant buffer (camera matrices, light, particle params, shading mode)
// DescriptorTable(SRV(t0, numDescriptors=3)): position, density, and LOD buffers, read-only for the vertex shader
// ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT is intentionally omitted: we no longer use a vertex buffer,
// the VS fetches particle positions directly from the structured buffer using SV_VertexID
#define ParticleRootSig "CBV(b0), DescriptorTable(SRV(t0, numDescriptors = 3))"

StructuredBuffer<float3> position : register(t0); // read-only view of the position buffer, indexed by SV_VertexID
StructuredBuffer<float>  density  : register(t1); // read-only view of the density buffer, indexed by SV_VertexID
StructuredBuffer<uint>   lod      : register(t2); // read-only view of the LOD buffer, indexed by SV_VertexID

#include "SharedConfig.hlsli"

struct LightData { float4 direction; float4 color; };

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat;
    float4x4 rayDirMat;
    float4 cameraPos;
    LightData lights[NUM_LIGHTS]; // must be here to match C++ layout
    float4 particleParams;
    uint shadingMode;
    uint minLOD;
    uint maxLOD;
    float _pad;
};

// VS passes the world-space position, density, and LOD to the geometry shader
struct VSOutput
{
    float3 worldPos : WORLDPOS; // particle center in world space, passed to GS for billboard generation
    float density : DENSITY;   // SPH density estimate, passed through for coloring in PS
    uint lod : LOD; // per-particle LOD value, passed through for coloring in PS
    uint particleIndex : PARTICLEINDEX; // index of this particle in the structured buffers
};

[RootSignature(ParticleRootSig)]
VSOutput main(uint vertexID : SV_VertexID) // SV_VertexID: auto-increments 0..N-1, one per DrawInstanced vertex, replaces IA vertex input
{
    VSOutput output;
    output.worldPos = position[vertexID]; // fetch this particle's world-space position from the structured buffer
    output.density  = density[vertexID];  // pass density through for visualization
    output.lod = lod[vertexID]; // pass LOD through for visualization
    output.particleIndex = vertexID; // pass particle index through
    return output;
}
