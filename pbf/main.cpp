#include "Egg/Common.h"
#include <Egg/App.h>
#include <Egg/Utility.h>
#include "PbfApp.h"
#include <imgui.h> // core ImGui functionality
#include <imgui_impl_win32.h> // ImGui's Win32 backend for handling window messages
#include <timeapi.h>
#pragma comment(lib, "winmm.lib") // for timeBeginPeriod / timeEndPeriod

// This is a forward declaration of a function that lives inside imgui_impl_win32.cpp. It's the Win32
// backend's message handler. We need to call it from our WindowProcess, but the header doesn't declare
// it directly(to avoid pulling in windows.h from ImGui headers), so we declare it ourselves with
// extern.We must forward Win32 messages to it so ImGui can track mouse position, clicks, keyboard input, etc.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

std::unique_ptr<PbfApp> app{ nullptr };

// callback function we'll register to the window for handling messages
LRESULT CALLBACK WindowProcess(HWND windowHandle, UINT message, WPARAM wParam, LPARAM lParam) {
    // let ImGui see every message first, so it can track mouse/keyboard state.
    // guard: after ShutdownImGui() destroys the context, Windows still delivers
    // a few trailing messages (e.g. WM_NCDESTROY). Without this check, those
    // would call into ImGui with a null context and trigger an assertion failure.
    if (ImGui::GetCurrentContext() != nullptr) {
        //ImGui_ImplWin32_WndProcHandler translates the
        // raw Win32 message into ImGui's internal input state — for example, WM_MOUSEMOVE updates io.MousePos,
        // WM_LBUTTONDOWN sets io.MouseDown[0] = true, WM_CHAR appends to the text input queue.
        //
        // If the handler returns true, it means ImGui fully consumed the message(e.g., the user is typing
        // into an InputFloat field — those characters should not also move the camera).We return immediately
        // without forwarding to our app.
        if (ImGui_ImplWin32_WndProcHandler(windowHandle, message, wParam, lParam))
            return true; 
    }

    switch (message) {
    case WM_DESTROY:// WM_DESTROY is sent when the user clicks the X button
		app->ShutdownImGui(); // release ImGui resources before destroying the D3D12 device
		app->Destroy(); // make sure to clean up our resources before exiting
        PostQuitMessage(0); // PostQuitMessage tells Windows to post WM_QUIT, which will break our message loop
        return 0;
	case WM_SIZE: // WM_SIZE is sent when the window is resized
        if (app) {
			int height = HIWORD(lParam); // the new height is stored in the high word of lParam
			int width = LOWORD(lParam); // the new width is stored in the low word of lParam
			app->Resize(width, height); // forward the new size to our app so it can resize its resources (e.g. swap chain)
        }
        break;
    }

    // This is the input arbitration logic — deciding whether our app or ImGui "owns" each message.
    // ImGui::GetIO() returns the IO structure where ImGui exposes two flags :
    // - WantCaptureMouse - true when the mouse is hovering over or interacting with an ImGui window
    // - WantCaptureKeyboard - true when an ImGui text field has focus
    // We check mouse and keyboard capture separately: hovering over the ImGui panel
    // should block mouse input but not keyboard input, and vice versa.
    // Key-up and mouse-up events are always forwarded regardless of capture state,
    // because if a key-down was forwarded but the matching key-up is blocked
    // (e.g. mouse drifted over ImGui between press and release), the camera's
    // "key held" flag would get stuck and movement would continue forever.
    if (app && ImGui::GetCurrentContext() != nullptr) {
        ImGuiIO& io = ImGui::GetIO();

        bool isKeyboard = (message >= WM_KEYFIRST && message <= WM_KEYLAST);
        bool isMouse = (message >= WM_MOUSEFIRST && message <= WM_MOUSELAST);
        bool isKeyUp = (message == WM_KEYUP || message == WM_SYSKEYUP);
        bool isMouseUp = (message == WM_LBUTTONUP || message == WM_RBUTTONUP || message == WM_MBUTTONUP);

        bool blocked = false;
        if (isKeyboard && !isKeyUp && io.WantCaptureKeyboard) blocked = true;
        if (isMouse && !isMouseUp && io.WantCaptureMouse) blocked = true;

        if (!blocked) {
            app->ProcessMessage(windowHandle, message, wParam, lParam);
        }
    }
    return DefWindowProcW(windowHandle, message, wParam, lParam); // for other messages: default behavior (mostly nothing)
}

HWND InitWindow(HINSTANCE hInstance) {
    // this string identifies our window class - it's an arbitrary name we pick,
    // used to connect the window class registration with the window creation below
    const wchar_t* windowClassName = L"PBFWindowClass";

    WNDCLASSW windowClass;
    ZeroMemory(&windowClass, sizeof(WNDCLASSW)); // zero out any memory trash
    windowClass.lpfnWndProc = WindowProcess; // assign our callback function to handle window messages
    windowClass.lpszClassName = windowClassName; // assign the class name - used later in CreateWindowExW
    windowClass.hInstance = hInstance; // the OS uses this to identify which program owns the window

	
    RegisterClassW(&windowClass); // register the window class with the OS before we make a window of that class

    HWND wnd = CreateWindowExW(
        0, // extended window style flags (none)
        windowClassName, // must match the class name we registered above
        L"Position Based Fluids", // the text that appears in the title bar
        WS_OVERLAPPEDWINDOW, // standard window style: title bar, border, minimize/maximize/close buttons
        CW_USEDEFAULT, // initial X position - let Windows decide
        CW_USEDEFAULT, // initial Y position - let Windows decide
        1280, 720, // window width and height in pixels (including title bar and borders)
        NULL, // no parent window
        NULL, // no menu
        hInstance, // same instance handle we used for the window class
        NULL);  // no extra creation data

	ASSERT(wnd != NULL, "Failed to create window"); // make sure window creation succeeded
    return wnd;
}

void InitDX12(
    com_ptr<ID3D12Debug>& debugController, 
    com_ptr<IDXGIFactory6>& dxgiFactory, 
    com_ptr<ID3D12Device>& device) {
    DX_API("Failed to create debug layer")
        D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()));
    //debugController->EnableDebugLayer(); // turn on the debug layer - causes D3D12 to validate API calls
    DX_API("Failed to create DXGI factory")
        CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf()));

    std::vector<com_ptr<IDXGIAdapter1>> adapters; // we'll store our enumerated GPUs here
    Egg::Utility::GetAdapters(dxgiFactory.Get(), adapters); // enumerate GPUs (e.g. integrated and dedicated graphics cards)
    ASSERT(!adapters.empty(), "No graphics adapters found");
    // pick the adapter with the most dedicated VRAM (typically the discrete GPU)
    int bestIndex = 0;
    SIZE_T bestVram = 0;
    for (int i = 0; i < (int)adapters.size(); i++) {
        DXGI_ADAPTER_DESC1 d;
        adapters[i]->GetDesc1(&d);
        if (d.DedicatedVideoMemory > bestVram) {
            bestVram = d.DedicatedVideoMemory;
            bestIndex = i;
        }
    }
    com_ptr<IDXGIAdapter1> tempAdapter = adapters[bestIndex];
    DXGI_ADAPTER_DESC1 desc;
    tempAdapter->GetDesc1(&desc);
    Egg::Utility::WDebugf(L"Selected adapter: %s, VRAM: %llu MB\n", desc.Description, desc.DedicatedVideoMemory / (1024 * 1024));
    IUnknown* selectedAdapter = tempAdapter.Get(); // untyped pointer for D3D12 device creation
    DX_API("Failed to create D3D Device")
        D3D12CreateDevice(selectedAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())); // try feature level 12?
}

void InitCommandQueue(com_ptr<ID3D12CommandQueue>& commandQueue, com_ptr<ID3D12Device> device) {
    // a command queue is where we submit work (draw calls, compute dispatches) for the GPU to execute
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc;
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; // DIRECT->general purpose: compute, copy, and draw
    commandQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL; // normal scheduling priority
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE; // no special flags
    commandQueueDesc.NodeMask = 0; // 0 = single GPU setup

    DX_API("Failed to create command queue")
        device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf())); // note that we create this via the d3d device

}

void InitSwapChain(
    com_ptr<IDXGISwapChain3>& swapChain,
    com_ptr<IDXGIFactory6> dxgiFactory,
    com_ptr<ID3D12CommandQueue> commandQueue,
    HWND windowHandle) {
    // swap chain creation: a set of (back)buffers we render into
    // while we're drawing to one, the other can be displayed on screen, then Present()-ed
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = { 0 }; // initializer syntax to zero out the struct, so we don't have to manually set every field to 0
    swapChainDesc.Width = 0; // 0 means query the size from the window
    swapChainDesc.Height = 0; // 0 means query the size from the window
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // 8 bits per channel RGBA, standard color format
    swapChainDesc.Stereo = false; // no stereoscopic 3D
    swapChainDesc.SampleDesc.Count = 1; // no multisampling, 1 sample per pixel
    swapChainDesc.SampleDesc.Quality = 0; // standard quality level
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; // we'll render into these buffers
    swapChainDesc.BufferCount = 2; // double buffering: 1 front buffer for display, 1 back buffer for rendering
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH; // stretch to fit the window 
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; // GPU discards old buffer after present
    swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE; // no blending, ignore alpha channel
    swapChainDesc.Flags = 0; // no special flags

    DXGI_SWAP_CHAIN_FULLSCREEN_DESC swapChainFullscreenDesc = { 0 }; // initializer syntax to zero out the struct, no manual set to 0
    swapChainFullscreenDesc.RefreshRate = DXGI_RATIONAL{ 60, 1 }; // 60 Hz refresh rate, 1 denominator means it's an integer
    swapChainFullscreenDesc.Windowed = true; // start in windowed mode
    swapChainFullscreenDesc.Scaling = DXGI_MODE_SCALING_CENTERED; // don't stretch in fullscreen, just center the image
    swapChainFullscreenDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UPPER_FIELD_FIRST; // standard scanline order

    // CreateSwapChainForHwnd gives us a SwapChain1, but we need SwapChain3
    // (for GetCurrentBackBufferIndex), so we create it first then cast.
    com_ptr<IDXGISwapChain1> tempSwapChain;
    DX_API("Failed to create swap chain for HWND")
        dxgiFactory->CreateSwapChainForHwnd( // created through dxgi factory, not d3d device!
            commandQueue.Get(), // the swap chain needs the command queue to know which GPU to present from
            windowHandle, // the window we're rendering into
            &swapChainDesc, // the swap chain description we filled out above
            &swapChainFullscreenDesc, // the fullscreen behavior description we filled out above
            NULL,// no restricting to a specific output/monitor
            tempSwapChain.GetAddressOf()); // store the created swap chain in our temporary com_ptr

    // cast from IDXGISwapChain1 to IDXGISwapChain3 so we can use GetCurrentBackBufferIndex()
    DX_API("Failed to cast swap chain")
        tempSwapChain.As(&swapChain); // .As() is a helper function that queries for the requested interface and casts if successful
}

std::unique_ptr<PbfApp> InitPbfApp(
    com_ptr<ID3D12Device> device,
    com_ptr<ID3D12CommandQueue> commandQueue,
    com_ptr<IDXGISwapChain3> swapChain,
    HWND windowHandle) {
	auto pbfApp = std::make_unique<PbfApp>();

    // set attributes
    pbfApp->SetDevice(device);
    pbfApp->SetCommandQueue(commandQueue);
    pbfApp->SetSwapChain(swapChain);

    // create and load resources, ordering is important here: some resources depend on others being created first
    pbfApp->CreateSwapChainResources(); // creates render target views and depth buffer
    pbfApp->CreateResources(); // creates fence, command allocator, command list
    pbfApp->LoadAssets();  // we'll load shaders and geometry here later
    pbfApp->InitImGui(windowHandle); // initialize Dear ImGui for the parameter tuning UI

    return pbfApp;
}

void RunMessageLoop() {
    // message loop: Windows communicates with our app by putting messages into a queue
    // keep pulling messages out and dispatching them until we get WM_QUIT -> stop
    MSG winMessage = { 0 };
    while (winMessage.message != WM_QUIT) {
        // PeekMessage checks if there's a message waiting, PM_REMOVE means: take the message out of the queue
        if (PeekMessage(&winMessage, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&winMessage);  // translates virtual-key messages into character messages
            DispatchMessage(&winMessage);   // sends the message to our WindowProcess callback
        }
        else {
            // No messages waiting: update and render.
            app->Run();
        }
    }
}

// wWinMain is the entry point for Windows GUI applications (the wide-char version of WinMain)
int APIENTRY wWinMain(
    _In_ HINSTANCE hInstance, // a handle to this running instance of the program
    _In_opt_ HINSTANCE hPrevInstance, // always NULL on modern Windows, legacy leftover
    _In_ LPWSTR command, // command line arguments as a wide string
    _In_ INT nShowCmd) // how the window should be shown (normal, minimized, maximized, etc.)
{
    HWND windowHandle = InitWindow(hInstance);

	com_ptr<ID3D12Debug> debugController{ nullptr }; // debug controller for detailer d3d12 error messages
	com_ptr<IDXGIFactory6> dxgiFactory{ nullptr }; // directx graphics infrastructure factory, used to create the swap chain and query video adapters
    com_ptr<ID3D12Device> device{ nullptr }; // D3D12 device for creating all D3D12 resources

	InitDX12(debugController, dxgiFactory, device); // initialize D3D12 and create the device

    com_ptr<ID3D12CommandQueue> commandQueue{ nullptr }; // we'll store the created command queue here
	InitCommandQueue(commandQueue, device); // create the command queue, which we'll use to submit work to the GPU

    com_ptr<IDXGISwapChain3> swapChain; // we'll store the created swap chain here
	InitSwapChain(swapChain, dxgiFactory, commandQueue, windowHandle);

    app = InitPbfApp(device, commandQueue, swapChain, windowHandle); // create our application instance, which will manage the rendering loop and resources
    app->RunNTimes(2, true); // run twice to populate all front and back buffers

    // Disable ALT+Enter fullscreen shortcut
    DX_API("Failed to make window association")
        dxgiFactory->MakeWindowAssociation(windowHandle, DXGI_MWA_NO_ALT_ENTER);

    ShowWindow(windowHandle, nShowCmd); // make the window visible - nShowCmd controls if it's normal/minimized/etc.

    timeBeginPeriod(1); // raise Windows timer resolution to 1ms
    RunMessageLoop();
	timeEndPeriod(1); // restore default timer resolution

    return 0;
}