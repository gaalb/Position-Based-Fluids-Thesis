#include "NeuralPostProcess.h"
#include <Egg/Shader.h>
#include <stdexcept>
#include <string>

NeuralPostProcess::P NeuralPostProcess::Create(
    ID3D12Device*       device,
    ID3D12CommandQueue* commandQueue,
    const char*         modelPath)
{
    auto p = std::shared_ptr<NeuralPostProcess>(new NeuralPostProcess());
    try {
        p->Init(device, commandQueue, modelPath);
        p->enabled = true;
    } catch (const std::exception& e) {
        OutputDebugStringA(("NeuralPostProcess::Create failed: " + std::string(e.what()) + "\n").c_str());
        // Return a disabled (non-null) instance so the rest of the app works normally.
    }
    return p;
}

NeuralPostProcess::~NeuralPostProcess() {
    FreeTensors();
}

// ---------------------------------------------------------------------------
// One-time initialization
// ---------------------------------------------------------------------------

void NeuralPostProcess::Init(ID3D12Device* device, ID3D12CommandQueue* commandQueue, const char* modelPath) {
    DX_API("Failed to create DML device")
        DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(dmlDevice.ReleaseAndGetAddressOf()));

    CreateDescHeap(device);
    CreateComputePipelines(device);
    CreateOrtSession(commandQueue, modelPath);
}

void NeuralPostProcess::CreateDescHeap(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC d = {};
    d.NumDescriptors = 4; // [0] sceneRT SRV, [1] dmlInput UAV, [2] dmlOutput SRV, [3] outputTex UAV
    d.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    d.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    DX_API("Failed to create neural descriptor heap")
        device->CreateDescriptorHeap(&d, IID_PPV_ARGS(descHeap.ReleaseAndGetAddressOf()));
    descSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void NeuralPostProcess::CreateComputePipelines(ID3D12Device* device) {
    auto toBlob   = Egg::Shader::LoadCso("Shaders/toNchwCS.cso");
    auto fromBlob = Egg::Shader::LoadCso("Shaders/fromNchwCS.cso");

    // Both shaders embed the identical NeuralRS root signature; extract from either.
    neuralRootSig = Egg::Shader::LoadRootSignature(device, toBlob.Get());

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = neuralRootSig.Get();

    psoDesc.CS = CD3DX12_SHADER_BYTECODE(toBlob.Get());
    DX_API("Failed to create toNchw PSO")
        device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(toNchwPso.ReleaseAndGetAddressOf()));

    psoDesc.CS = CD3DX12_SHADER_BYTECODE(fromBlob.Get());
    DX_API("Failed to create fromNchw PSO")
        device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(fromNchwPso.ReleaseAndGetAddressOf()));
}

void NeuralPostProcess::CreateOrtSession(ID3D12CommandQueue* commandQueue, const char* modelPath) {
    const OrtApi& api = Ort::GetApi();

    // Acquire the DML extension API pointer.
    Ort::ThrowOnError(api.GetExecutionProviderApi(
        "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi)));

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.DisableMemPattern();       // required for DML EP
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    // V2 DML EP: use our D3D12 device (via dmlDevice) and our graphics commandQueue.
    // ORT submits DML command lists to commandQueue, so inference is FIFO after our draws.
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML1(
        opts, dmlDevice.Get(), commandQueue));

    // ORT 1.20+ dropped the const char* Session ctor; convert to wide string.
    std::wstring wpath(modelPath, modelPath + std::strlen(modelPath));
    ortSession = std::make_unique<Ort::Session>(ortEnv, wpath.c_str(), opts);
}

// ---------------------------------------------------------------------------
// Resolution-dependent resources
// ---------------------------------------------------------------------------

void NeuralPostProcess::Resize(ID3D12Device* device, UINT w, UINT h) {
    ReleaseResolutionResources();
    width = w; height = h;
    CreateSceneRT(device);
    CreateDmlBuffers(device);
    CreateOutputTex(device);
    RefreshDescriptors(device);
    if (ortSession) RebindTensors();
}

void NeuralPostProcess::ReleaseResolutionResources() {
    FreeTensors();
    ioBinding.reset();
    sceneRT.Reset();
    dmlInputBuf.Reset();
    dmlOutputBuf.Reset();
    outputTex.Reset();
}

void NeuralPostProcess::CreateSceneRT(ID3D12Device* device) {
    D3D12_CLEAR_VALUE cv = {};
    cv.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    cv.Color[1] = 0.2f; cv.Color[2] = 0.4f; cv.Color[3] = 1.0f; // match scene clear color

    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    DX_API("Failed to create neural scene RT")
        device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, &cv,
            IID_PPV_ARGS(sceneRT.ReleaseAndGetAddressOf()));
    sceneRT->SetName(L"Neural SceneRT");

    // RTV heap is allocated once; the view is recreated each Resize.
    if (!sceneRtvHeap) {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 1;
        hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        DX_API("Failed to create scene RTV heap")
            device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(sceneRtvHeap.ReleaseAndGetAddressOf()));
    }
    sceneRtvHandle = sceneRtvHeap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(sceneRT.Get(), nullptr, sceneRtvHandle);
}

void NeuralPostProcess::CreateDmlBuffers(ID3D12Device* device) {
    UINT64 inBytes  = (UINT64)4 * height * width * sizeof(float); // [1,4,H,W]
    UINT64 outBytes = (UINT64)3 * height * width * sizeof(float); // [1,3,H,W]
    auto mkBuf = [](UINT64 size) {
        return CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    };
    DX_API("Failed to create dmlInputBuf")
        device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &mkBuf(inBytes), D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(dmlInputBuf.ReleaseAndGetAddressOf()));
    dmlInputBuf->SetName(L"DML Input Buffer");

    DX_API("Failed to create dmlOutputBuf")
        device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &mkBuf(outBytes), D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(dmlOutputBuf.ReleaseAndGetAddressOf()));
    dmlOutputBuf->SetName(L"DML Output Buffer");
}

void NeuralPostProcess::CreateOutputTex(ID3D12Device* device) {
    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    DX_API("Failed to create neural output texture")
        device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(outputTex.ReleaseAndGetAddressOf()));
    outputTex->SetName(L"Neural OutputTex");
}

void NeuralPostProcess::RefreshDescriptors(ID3D12Device* device) {
    auto cpu = descHeap->GetCPUDescriptorHandleForHeapStart();

    // [0] sceneRT SRV (Texture2D<float4>)
    D3D12_SHADER_RESOURCE_VIEW_DESC rtSrv = {};
    rtSrv.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtSrv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    rtSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    rtSrv.Texture2D.MipLevels     = 1;
    device->CreateShaderResourceView(sceneRT.Get(), &rtSrv, cpu);
    cpu.ptr += descSize;

    // [1] dmlInputBuf UAV (RWBuffer<float>, R32_FLOAT typed)
    D3D12_UNORDERED_ACCESS_VIEW_DESC inUav = {};
    inUav.Format              = DXGI_FORMAT_R32_FLOAT;
    inUav.ViewDimension       = D3D12_UAV_DIMENSION_BUFFER;
    inUav.Buffer.NumElements  = 4 * height * width;
    device->CreateUnorderedAccessView(dmlInputBuf.Get(), nullptr, &inUav, cpu);
    cpu.ptr += descSize;

    // [2] dmlOutputBuf SRV (Buffer<float>, R32_FLOAT typed)
    D3D12_SHADER_RESOURCE_VIEW_DESC outSrv = {};
    outSrv.Format                  = DXGI_FORMAT_R32_FLOAT;
    outSrv.ViewDimension           = D3D12_SRV_DIMENSION_BUFFER;
    outSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    outSrv.Buffer.NumElements      = 3 * height * width;
    device->CreateShaderResourceView(dmlOutputBuf.Get(), &outSrv, cpu);
    cpu.ptr += descSize;

    // [3] outputTex UAV (RWTexture2D<unorm float4>, R8G8B8A8_UNORM)
    D3D12_UNORDERED_ACCESS_VIEW_DESC texUav = {};
    texUav.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
    texUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(outputTex.Get(), nullptr, &texUav, cpu);
}

void NeuralPostProcess::RebindTensors() {
    FreeTensors();

    // The model was exported with fixed H×W. If the current resolution doesn't match,
    // disable the pass rather than binding mismatched tensors to a fixed-shape model.
    auto inInfo  = ortSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto inDims  = inInfo.GetShape(); // [1, 4, H, W]
    if (inDims.size() == 4 && inDims[2] > 0 && inDims[3] > 0) {
        modelWidth  = (UINT)inDims[3];
        modelHeight = (UINT)inDims[2];
        if (modelHeight != height || modelWidth != width) {
            OutputDebugStringA(("NeuralPostProcess: resolution mismatch — model is " +
                std::to_string(modelWidth) + "x" + std::to_string(modelHeight) +
                ", render is " + std::to_string(width) + "x" + std::to_string(height) +
                ". Re-export with --width " + std::to_string(width) +
                " --height " + std::to_string(height) + "\n").c_str());
            enabled = false;
            return;
        }
    }

    // Wrap our D3D12 buffers as DML GPU memory for ORT tensor binding.
    Ort::ThrowOnError(dmlApi->CreateGPUAllocationFromD3DResource(dmlInputBuf.Get(),  &inputDmlAlloc));
    Ort::ThrowOnError(dmlApi->CreateGPUAllocationFromD3DResource(dmlOutputBuf.Get(), &outputDmlAlloc));

    Ort::MemoryInfo dmlMem("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

    int64_t inShape[]  = { 1, 4, (int64_t)height, (int64_t)width };
    int64_t outShape[] = { 1, 3, (int64_t)height, (int64_t)width };
    size_t  inBytes    = (size_t)4 * height * width * sizeof(float);
    size_t  outBytes   = (size_t)3 * height * width * sizeof(float);

    inputTensor  = Ort::Value::CreateTensor(dmlMem, inputDmlAlloc,  inBytes,  inShape,  4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    outputTensor = Ort::Value::CreateTensor(dmlMem, outputDmlAlloc, outBytes, outShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Bind once; the same buffers are reused every frame.
    ioBinding = std::make_unique<Ort::IoBinding>(*ortSession);
    ioBinding->BindInput("frame_rgba",   inputTensor);
    ioBinding->BindOutput("enhanced_rgb", outputTensor);
}

void NeuralPostProcess::FreeTensors() {
    ioBinding.reset();
    inputTensor  = Ort::Value{ nullptr };
    outputTensor = Ort::Value{ nullptr };
    if (dmlApi) {
        if (inputDmlAlloc)  { dmlApi->FreeGPUAllocation(inputDmlAlloc);  inputDmlAlloc  = nullptr; }
        if (outputDmlAlloc) { dmlApi->FreeGPUAllocation(outputDmlAlloc); outputDmlAlloc = nullptr; }
    }
}

// ---------------------------------------------------------------------------
// Per-frame recording
// ---------------------------------------------------------------------------

void NeuralPostProcess::RecordPreProcess(ID3D12GraphicsCommandList* cmd) {
    // sceneRT: RENDER_TARGET → NON_PIXEL_SHADER_RESOURCE (toNchwCS reads it as Texture2D SRV)
    Transition(cmd, sceneRT.Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // dmlInputBuf: COMMON → UNORDERED_ACCESS (toNchwCS writes via RWBuffer<float>)
    Transition(cmd, dmlInputBuf.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Bind neural heap and dispatch toNchwCS.
    // Root param 0: descriptor table [sceneRT SRV @ slot 0, dmlInputBuf UAV @ slot 1]
    // Root param 1: {W, H} root constants
    ID3D12DescriptorHeap* heaps[] = { descHeap.Get() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(neuralRootSig.Get());
    cmd->SetPipelineState(toNchwPso.Get());
    cmd->SetComputeRootDescriptorTable(0,
        CD3DX12_GPU_DESCRIPTOR_HANDLE(descHeap->GetGPUDescriptorHandleForHeapStart(), 0, descSize));
    UINT dims[2] = { width, height };
    cmd->SetComputeRoot32BitConstants(1, 2, dims, 0);
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);

    // UAV barrier to flush toNchwCS writes, then hand dmlInputBuf back to COMMON for ORT.
    cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(dmlInputBuf.Get()));
    Transition(cmd, dmlInputBuf.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

    // Restore sceneRT to RENDER_TARGET for the next frame's PrepareCommandList.
    Transition(cmd, sceneRT.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
}

void NeuralPostProcess::RunInference() {
    // ORT records DML operator dispatches and submits them to our commandQueue.
    // Because it's the same FIFO queue, inference runs strictly after phase-1 work.
    try{
        ortSession->Run(Ort::RunOptions{}, *ioBinding);
}
    catch (const Ort::Exception& e) {
        // Handle exception
		e.GetOrtErrorCode(); // Get the error code
    }
}

void NeuralPostProcess::RecordPostProcess(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource*            backbuffer)
{
    // dmlOutputBuf: ORT leaves it in COMMON after writing; transition to SRV for fromNchwCS.
    Transition(cmd, dmlOutputBuf.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // outputTex is in UNORDERED_ACCESS (from creation or previous RecordPostProcess exit).

    // Bind neural heap and dispatch fromNchwCS.
    // Root param 0: descriptor table [dmlOutputBuf SRV @ slot 2, outputTex UAV @ slot 3]
    // Root param 1: {W, H} root constants
    ID3D12DescriptorHeap* heaps[] = { descHeap.Get() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(neuralRootSig.Get());
    cmd->SetPipelineState(fromNchwPso.Get());
    cmd->SetComputeRootDescriptorTable(0,
        CD3DX12_GPU_DESCRIPTOR_HANDLE(descHeap->GetGPUDescriptorHandleForHeapStart(), 2, descSize));
    UINT dims[2] = { width, height };
    cmd->SetComputeRoot32BitConstants(1, 2, dims, 0);
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);

    // Flush fromNchwCS writes; outputTex → COPY_SOURCE for CopyResource.
    cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(outputTex.Get()));
    Transition(cmd, outputTex.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

    // dmlOutputBuf → COMMON (ready for ORT next frame).
    Transition(cmd, dmlOutputBuf.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);

    // Backbuffer: PRESENT → COPY_DEST, copy, → RENDER_TARGET for ImGui overlay.
    Transition(cmd, backbuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd->CopyResource(backbuffer, outputTex.Get());
    Transition(cmd, backbuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);

    // Restore outputTex → UNORDERED_ACCESS for next frame's fromNchwCS.
    Transition(cmd, outputTex.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void NeuralPostProcess::Transition(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource*            res,
    D3D12_RESOURCE_STATES      from,
    D3D12_RESOURCE_STATES      to)
{
    if (from == to) return;
    auto b = CD3DX12_RESOURCE_BARRIER::Transition(res, from, to);
    cmd->ResourceBarrier(1, &b);
}
