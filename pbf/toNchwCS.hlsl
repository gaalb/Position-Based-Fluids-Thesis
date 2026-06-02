// Converts the rendered scene texture (RGBA8) into a float32 NCHW buffer
// for DirectML inference. One thread per pixel; layout: channel * H * W + y * W + x.
// Bind: t0 = scene RT (SRV), u0 = DML input buffer (UAV), b0 = {W, H} root constants.
// Dispatch: (ceil(W/8), ceil(H/8), 1)

#define NeuralRS \
    "RootFlags(0), " \
    "DescriptorTable(SRV(t0, numDescriptors=1), UAV(u0, numDescriptors=1)), " \
    "RootConstants(num32BitConstants=2, b0)"

Texture2D<float4> gSceneRT  : register(t0);
RWBuffer<float>   gDmlInput : register(u0);

cbuffer Dims : register(b0) { uint gW; uint gH; }

[RootSignature(NeuralRS)]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x, y = tid.y;
    if (x >= gW || y >= gH) return;
    float4 rgba = gSceneRT[uint2(x, y)];
    uint base   = y * gW + x;
    uint stride = gH * gW;
    gDmlInput[0u * stride + base] = rgba.r;
    gDmlInput[1u * stride + base] = rgba.g;
    gDmlInput[2u * stride + base] = rgba.b;
    gDmlInput[3u * stride + base] = rgba.a;
}
