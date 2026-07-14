#define ParticleRootSig "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0)"

struct VSOutput
{
    float3 worldPos : WORLDPOS; // the world position of the vertex, passed to GS for billboard generation
    float density : DENSITY;   // SPH density estimate, passed through for coloring in PS
    uint lod : LOD;            // per-particle LOD value, passed through for coloring in PS
    uint particleIndex : PARTICLEINDEX; // index of this particle in the structured buffers
};

struct GSOutput
{
    float4 position : SV_Position; // clip-space position for rasterization
    float2 uv : TEXCOORD; // ranges from (-1,-1) to (1,1), center is (0,0)
    float3 centerWorld : CENTERW; // particle center in world space (same for all 4 verts)
    float3 right : RIGHT; // billboard right axis in world space
    float3 up : UP; // billboard up axis in world space
    float density : DENSITY; // SPH density, same for all 4 verts of this billboard
    uint lod : LOD;          // LOD value, same for all 4 verts of this billboard
    uint particleIndex : PARTICLEINDEX; // index of this particle in the structured buffers    
};

#include "SharedConfig.hlsli"

struct LightData { float4 direction; float4 color; };

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat;
    float4x4 rayDirMat;
    float4 cameraPos;
    LightData lights[NUM_LIGHTS]; // must be here to match C++ layout
    float4 particleParams;        // w = particle display radius
};

// Each input point produces 4 vertices (a triangle strip forming a quad)
[RootSignature(ParticleRootSig)]
[maxvertexcount(4)]
void main(point VSOutput input[1], inout TriangleStream<GSOutput> outputStream)
{
    float3 center = input[0].worldPos;
    float radius = particleParams.w;

    // compute billboard axes: the quad always faces the camera
    float3 forward = normalize(cameraPos.xyz - center); // points particle -> camera
    float3 worldUp = float3(0, 1, 0); // we're supposing the camera always stands upright
    float3 right = normalize(cross(worldUp, forward)); // billboard's local right
    float3 up = normalize(cross(forward, right)); // billboard's local up

    // che 4 corners of the quad, offset from center by +-right and +-up
    // UV coordinates map from -1 to 1 so we can compute sphere normals in the PS
    float2 uvs[4] =
    {
        float2(-1, -1), // bottom-left
        float2(1, -1), // bottom-right
        float2(-1, 1), // top-left
        float2(1, 1) // top-right
    };

    // emit 4 vertices as a triangle strip: forms 2 triangles (bottom-left, bottom-right, top-left, top-right)
    for (int i = 0; i < 4; i++)
    {
        GSOutput output;
        output.particleIndex = input[0].particleIndex;

        // offset the corner from center in world space
        float3 worldPos = center + right * uvs[i].x * radius + up * uvs[i].y * radius;

        // project to clip space
        output.position = mul(viewProjMat, float4(worldPos, 1.0));
        output.uv = uvs[i];
        output.centerWorld = center;
        output.right = right;
        output.up = up;
        output.density = input[0].density;
        output.lod     = input[0].lod;

        outputStream.Append(output);
    }
}