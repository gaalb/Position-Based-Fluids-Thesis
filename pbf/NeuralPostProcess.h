#pragma once
#include <Egg/Common.h>
#include <Egg/d3dx12.h>
#include <DirectML.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <memory>

// Neural post-processing pass: runs a U-Net on the rendered frame using
// DirectML via ONNX Runtime.
//
// Integration contract (two-phase per frame):
//   Phase 1  — scene draws to GetSceneRT(), then RecordPreProcess(), close + execute cmdList
//   Inference — RunInference() submits DML work to the same commandQueue (FIFO after phase 1)
//   Phase 2  — fresh Reset() on cmdList, RecordPostProcess(), then ImGui, then ExecuteGraphics()
//
// Required packages (extract to Common\onnxruntime\ and Common\directml\):
//   Microsoft.ML.OnnxRuntime.DirectML  (NuGet) — onnxruntime.lib, DirectML.dll, headers
//   Microsoft.AI.DirectML              (NuGet) — DirectML.h, DirectML.lib
//
// The ONNX model must be at Bin\models\unet_postprocess.onnx at runtime.
// Generate it with: python models/export_unet.py
class NeuralPostProcess {
public:
    using P = std::shared_ptr<NeuralPostProcess>;

    // commandQueue: the graphics DIRECT queue — ORT submits DML work here (FIFO after our draws).
    static P Create(ID3D12Device* device, ID3D12CommandQueue* commandQueue, const char* modelPath);

    ~NeuralPostProcess();

    // Recreate all resolution-dependent resources. Call from CreateSwapChainResources and Resize.
    void Resize(ID3D12Device* device, UINT width, UINT height);

    // Release resolution-dependent resources. Call from ReleaseSwapChainResources.
    void ReleaseResolutionResources();

    // Scene draws go here instead of the swapchain backbuffer.
    ID3D12Resource*             GetSceneRT()        const { return sceneRT.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE GetSceneRtvHandle() const { return sceneRtvHandle; }

    // Phase 1: dispatch toNchwCS (sceneRT → dmlInputBuf), then barrier + transition dmlInputBuf to
    // COMMON for ORT. Leaves sceneRT in RENDER_TARGET (ready for the next frame's PrepareCommandList).
    // Call after all scene draws; cmdList must be open and recording.
    void RecordPreProcess(ID3D12GraphicsCommandList* cmdList);

    // Submit ORT inference to commandQueue. FIFO after phase 1. Returns before GPU finishes.
    void RunInference();

    // Phase 2: dispatch fromNchwCS (dmlOutputBuf → outputTex), then:
    //   backbuffer: PRESENT → COPY_DEST  (CopyResource outputTex→backbuffer) → RENDER_TARGET
    // The caller can draw ImGui on the backbuffer (RENDER_TARGET) after this returns.
    void RecordPostProcess(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* backbuffer);

    bool enabled = false;
    UINT modelWidth = 0, modelHeight = 0;  // expected resolution from the ONNX model

private:
    NeuralPostProcess() = default;

    UINT width = 0, height = 0;

    // Intermediate render target (R8G8B8A8_UNORM, ALLOW_RENDER_TARGET).
    // Scene draws here instead of the swapchain backbuffer.
    com_ptr<ID3D12Resource>         sceneRT;
    com_ptr<ID3D12DescriptorHeap>   sceneRtvHeap;
    D3D12_CPU_DESCRIPTOR_HANDLE     sceneRtvHandle = {};

    // DML I/O buffers (typed R32_FLOAT, ALLOW_UNORDERED_ACCESS).
    // dmlInputBuf:  [1, 4, H, W] float32  — written by toNchwCS, read by ORT
    // dmlOutputBuf: [1, 3, H, W] float32  — written by ORT, read by fromNchwCS
    com_ptr<ID3D12Resource> dmlInputBuf;
    com_ptr<ID3D12Resource> dmlOutputBuf;

    // Output texture (R8G8B8A8_UNORM, ALLOW_UNORDERED_ACCESS).
    // fromNchwCS writes here; then CopyResource copies it to the swapchain backbuffer.
    com_ptr<ID3D12Resource> outputTex;

    // Shader-visible descriptor heap (4 slots):
    //  [0] sceneRT SRV     → t0 in toNchwCS
    //  [1] dmlInputBuf UAV → u0 in toNchwCS
    //  [2] dmlOutputBuf SRV→ t0 in fromNchwCS
    //  [3] outputTex UAV   → u0 in fromNchwCS
    com_ptr<ID3D12DescriptorHeap> descHeap;
    UINT descSize = 0;

    // Compute pipelines. Both shaders embed the same root signature (NeuralRS).
    com_ptr<ID3D12RootSignature> neuralRootSig;
    com_ptr<ID3D12PipelineState> toNchwPso;
    com_ptr<ID3D12PipelineState> fromNchwPso;

    // DirectML device (created from the app's ID3D12Device; needed for the V2 DML EP).
    com_ptr<IDMLDevice> dmlDevice;

    // ONNX Runtime (session created once; tensors recreated on Resize).
    Ort::Env                      ortEnv{ ORT_LOGGING_LEVEL_WARNING, "pbf_neural" };
    std::unique_ptr<Ort::Session> ortSession;
    std::unique_ptr<Ort::IoBinding> ioBinding;
    const OrtDmlApi*              dmlApi = nullptr;

    // GPU allocations wrapping dmlInputBuf / dmlOutputBuf for ORT tensor binding.
    void*      inputDmlAlloc  = nullptr;
    void*      outputDmlAlloc = nullptr;
    Ort::Value inputTensor{ nullptr };
    Ort::Value outputTensor{ nullptr };

    // One-time setup (device/session/pipelines). Called from Create().
    void Init(ID3D12Device* device, ID3D12CommandQueue* commandQueue, const char* modelPath);
    void CreateDescHeap(ID3D12Device* device);
    void CreateComputePipelines(ID3D12Device* device);
    void CreateOrtSession(ID3D12CommandQueue* commandQueue, const char* modelPath);

    // Resolution-dependent setup. Called from Resize().
    void CreateSceneRT(ID3D12Device* device);
    void CreateDmlBuffers(ID3D12Device* device);
    void CreateOutputTex(ID3D12Device* device);
    void RefreshDescriptors(ID3D12Device* device);
    void RebindTensors();
    void FreeTensors();

    static void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res,
                           D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to);
};
