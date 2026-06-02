// Converts a float32 NCHW buffer (DirectML output, RGB channels) into an
// RGBA8 UAV texture for final display. One thread per pixel.
// Bind: t0 = DML output buffer (SRV), u0 = output RGBA8 texture (UAV), b0 = {W, H} root constants.
// Dispatch: (ceil(W/8), ceil(H/8), 1)

#define NeuralRS \
    "RootFlags(0), " \
    "DescriptorTable(SRV(t0, numDescriptors=1), UAV(u0, numDescriptors=1)), " \
    "RootConstants(num32BitConstants=2, b0)"

Buffer<float>               gDmlOutput : register(t0);
RWTexture2D<unorm float4>   gOutputTex : register(u0);

cbuffer Dims : register(b0) { uint gW; uint gH; }

[RootSignature(NeuralRS)]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x, y = tid.y;
    if (x >= gW || y >= gH) return;
    uint base   = y * gW + x;
    uint stride = gH * gW;
    float r = gDmlOutput[0u * stride + base];
    float g = gDmlOutput[1u * stride + base];
    float b = gDmlOutput[2u * stride + base];
    gOutputTex[uint2(x, y)] = float4(saturate(r), saturate(g), saturate(b), 1.0f);
}
