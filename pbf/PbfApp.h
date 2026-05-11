#pragma once

#include <algorithm>
#include "AsyncComputeApp.h"
#include "ConstantBufferTypes.h"
#include "ParticleTypes.h"
#include "ComputeShader.h"
#include <Egg/Cam/FirstPerson.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/TextureCube.h>
#include <Egg/Importer.h>
#include <Egg/Mesh/Prefabs.h>
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>
#include "SolidObstacle.h"
#include "DescriptorAllocator.h"
#include "GpuBuffer.h"
#include "GpuTexture.h"
#include "DoubleBufferGpuBuffer.h"
#include "DoubleBufferGpuTexture.h"
#include "SpatialGrid.h"
#include "LodSystem.h"
#include <immintrin.h>
#include <thread>
#include "SharedConfig.hlsli" 

// CPU-side staging struct for ease of particle initialization.
struct ParticleInitData {
	std::vector<Float3> positions;
	std::vector<Float3> velocities;
};

// PBF implementation: based on the Macklin & Muller 2013 nvidia research article (M&M)
// AsyncComputeApp gives us:
// graphics command allocator and command list
// compute command queue, allocator, and command list
// depth stencil buffer
// PSO manager
// per-queue frame counters (graphicsFrame, computeFrame) and fences (graphicsFence, computeFence)
// sync helpers: cpuWaitForGraphics, cpuWaitForCompute, graphicsWaitForCompute
// note: we're not using graphicsFrame and computeFrame, instead, we use a single frame counter,
// because the two queues are currently in lockstep (this might change later if we want a different
// display and physics frequency)
//
// main calls app->Run() in an infinite loop, which calculates elapsedTime and deltaTime,
// and calls Update (overridden) and Render (overridden), in that order.
//
// Initialization flow:
// main.cpp first creates the window handle, command queue, d3d12 device and swap chain,
// then initializes the pbf App by calling these methods:
// SetDevice / SetCommandQueue / SetSwapChain
// CreateSwapChainResources: creates RTV heap and render target views for the swap chain back buffers
// CreateResources: creates command allocators and lists for both queues, PSO manager, fences, etc.
// LoadAssets: uploads initial data to the GPU and builds rendering/compute pipelines.
// InitImGui: sets up the ImGui context and its Win32 + D3D12 backends
// RunNTimes: runs the simulation loop a few times to populate all double buffers

class PbfApp : public AsyncComputeApp {
protected:
	// Fixed particle and grid constants.
	const int particlesX = 100, particlesY = 40, particlesZ = 100; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 30, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = particlesX * particlesY * particlesZ; // total number of particles in the simulation
	// particleSpacing and hMultiplier are constants that define the SPH kernel width h,
	// which gives a lower bound to the spatial grid's cell width. We can use (try using...)
	// Morton codes to index the cells, which works best with a cubic simulation space that
	// has a power of two number of cells along each axis. Fixing h (= particleSpacing * hMultiplier)
	// lets us define the box as exactly gridDim * h on each axis, giving a perfectly aligned
	// cubic grid with a dense Morton code space.
	const float particleSpacing = PARTICLE_SPACING; // inter-particle distance (also determines rest density and display size)
	const float hMultiplier = H_MULTIPLIER; // h = particleSpacing * hMultiplier
	const float h = H; // SPH smoothing radius
	// if the particles are spaced "d" apart, then one d sided cube contains one particle, meaning that
	// each particle is responsible for d^3 volume of fluid, meaning that with m=1, the density is 1/d^3
	const float rho0 = RHO0;
	const float externalAcceleration = 20.0f; // m/s^2, applied horizontally via arrow keys
	// Spatial grid: cubic, power-of-two cells per axis, box derived from grid.
	// The grid can use Morton code (Z-order curve) indexing, which requires equal power-of-two
	// dimensions on all axes for a dense index space (no wasted codes). We choose a single gridDim
	// for all three axes (cubic grid), such that the box has gridDim * h cells per axis.
	// With gridDim = 32 and h = 0.875, the box is 28 units per side, centered at the origin.
	const UINT gridDim = GRID_DIM; // cells per axis (must be power of two)
	const UINT numCells = gridDim * gridDim * gridDim; // total cells in the grid
	const float boxExtent = gridDim * h / CELL_PER_H; // box side length: gridDim cells of width h/CELL_PER_H
	const Float3 gridMin = Float3(-boxExtent / 2.0f, -boxExtent / 2.0f, -boxExtent / 2.0f); // most negative point of the grid
	const Float3 gridMax = Float3( boxExtent / 2.0f,  boxExtent / 2.0f,  boxExtent / 2.0f); // most positive point of the grid
	Float3 boxMin = Float3(-24.0f, -30.0f, -24.0f); // adjustable collision boundary
	Float3 boxMax = Float3(24.0f, 40, 24.0f); // adjustable collision boundary

	// parameters that are tunable via ImGui each frame
	int solverIterations = 6; // how many newton steps to take per frame
	int minLOD = 3;  // minimum solver iterations for the farthest particles
	float epsilon = 4.0f; // constraint force mixing relaxation parameter, higher value = softer constraints
	float viscosity = 0.005f; // // XSPH viscosity coefficient, higher value = "thicker" fluid, M&M: 0.01
	// artificial purely repulsive pressure term reduces clumping while leaving room for surface tension,
	float sCorrK = 0.02f; // artificial pressure magnitude coefficient M&M: 0.1
	float vorticityEpsilon = 0.01f; // vorticity confinement strength M&M: 0.01
	float adhesion = 0.01f; // tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
	bool fountainEnabled = false; // toggle for the upward jet in a corner of the box, like a fountain :)

	Float3 externalForce = Float3(0.0f, 0.0f, 0.0f); // current external acceleration from arrow keys
	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera
	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame -> graphics data
	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable
	Egg::Mesh::Shaded::P backgroundMesh; // fullscreen quad + cubemap shader
	Egg::TextureCube envTexture; // the cubemap texture for the skybox background

	Egg::ConstantBuffer<ComputeCb> computeCb; // constant buffer uploaded to GPU each frame -> compute data

	// Upload buffers for initial particle data (upload heap, CPU-writable).
	// Only position and velocity have meaningful initial values; other fields are
	// overwritten by the simulation before they're first read.
	GpuBuffer::P positionUploadBuffer;
	GpuBuffer::P velocityUploadBuffer;

	DescriptorAllocator::P imguiAllocator; // 1-slot shader-visible heap for ImGui font texture
	DescriptorAllocator::P mainAllocator; // shader-visible CBV/SRV/UAV heap
	DescriptorAllocator::P staticAllocator; // CPU-only non-shader-visible heap for CopyDescriptorsSimple sources

	// Particle field double buffers: front = live data (compute reads), back = sort target (permutateCS writes).
	// flip() is called once per frame (after cpuWaitForCompute) to promote the sorted back to front.
	DoubleBufferGpuBuffer::P particleFieldDB[PF_COUNT];

	SpatialGrid::P spatialGrid; // owns grid buffers and all grid/sort shaders
	LodSystem::P lod; // owns LOD buffers, depth textures, and all LOD shaders

	// One ComputeShader per pass. Each holds its own PSO, root signature, fixed descriptor
	// table handle, and input/output resource lists for UAV barrier insertion.
	ComputeShader::P predictShader; // apply forces + velocity collision response + predict p* in one per-particle pass
	ComputeShader::P collisionPredictedPositionShader; // clamp p* to box; runs pre-sort and every solver iteration
	ComputeShader::P lambdaShader; // compute lambda per particle
	ComputeShader::P deltaShader; // compute delta_p, write to scratch
	ComputeShader::P positionFromScratchShader; // copy scratch -> predictedPosition (Jacobi commit during solver loop)
	ComputeShader::P updateVelocityShader;// update velocity from displacement: v = (p* - x) / dt
	ComputeShader::P vorticityShader; // estimate per-particle vorticity (curl of velocity), store in omega
	ComputeShader::P confinementViscosityShader; // vorticity confinement + XSPH viscosity in one neighbor pass
	// GSM-boosted counterparts loaded alongside the plain shaders; dispatched when gsmEnabled is true.
	ComputeShader::P gsmLambdaShader;
	ComputeShader::P gsmDeltaShader;
	ComputeShader::P gsmVorticityShader;
	ComputeShader::P gsmConfinementViscosityShader;
	ComputeShader::P velocityFromScratchShader; // copy scratch -> velocity (Jacobi commit after viscosity)
	ComputeShader::P updatePositionShader; // update position from predictedPosition (final step per paper)

	// Solid obstacles: each owns a renderable mesh, SDF volume, and its own transform state.
	// To add a new obstacle: bump NUM_OBSTACLES in SharedConfig.hlsli, add an entry to
	// the ObstacleDesc table in InitObstacle(), and add the corresponding asset files.
	SolidObstacle::P obstacles[NUM_OBSTACLES];
	int selectedObstacle = 0; // which obstacle the ImGui sliders currently target

	// Directional light sources. To add a light: bump NUM_LIGHTS in SharedConfig.hlsli
	// and add an entry to each initializer list below.
	Float3 lightDirs[NUM_LIGHTS] = {
		Float3( 0.5f,  1.0f,  0.3f), // above-left-front, white
		Float3(-1.0f,  0.4f,  0.2f), // from the right, warm orange
		Float3( 0.2f, -0.3f, -1.0f), // from behind-below, cool blue
	};
	Float3 lightColors[NUM_LIGHTS] = {
		Float3(1.00f, 1.00f, 1.00f), // white
		Float3(1.00f, 0.75f, 0.40f), // warm orange, reduced intensity
		Float3(0.40f, 0.60f, 1.00f), // cool blue, reduced intensity
	};
	int selectedLight = 0;

	// Readback buffers (readback heap, CPU-readable after CopyBufferRegion)
	GpuBuffer::P densityReadbackBuffer;
	std::vector<float> densityReadbackData;
	float avgDensity = 0.0f; // average particle density from previous frame's readback
	GpuBuffer::P lodReadbackBuffer;
	std::vector<uint32_t> lodReadbackData;
	float avgLod = 0.0f; // average per-particle LOD from previous frame's readback

	// debug/throttle timers
	using clock = std::chrono::high_resolution_clock;
	clock::time_point lastFrame; // tracks accumulated time toward next physics step
	const float targetFps = 30.0f; // adjust this for different fps cap
	std::chrono::duration<double> targetPeriod{ 1.0 / targetFps }; // 60 fps cap
	clock::time_point t0, t1; // debug timer variabeles
	float debugTimer = 0.0f;
	float lastDt = 0.0f; // dt from last Update(), consumed by Render() for compute CB upload after GPU sync

	uint64_t frameCount = 0; // indexes each frame
	bool fpsCapped = false; // toggled from ImGui
	bool gsmEnabled = false; // use GSM-boosted shaders for lambda, delta, vorticity, confinementViscosity
	bool physicsRunning = false; // toggled by spacebar: when false, compute passes are skipped each frame

	// arrow key held state for external acceleration input
	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false;

	// Async compute: physics runs on the compute queue (from AsyncComputeApp), decoupled from vsync.
	// Double-buffered snapshot buffers hold position, density, and LOD for the graphics queue.
	// Compute writes to DB->getBack(); flip() (called after cpuWaitForCompute) promotes it to front for graphics.
	DoubleBufferGpuBuffer::P positionSnapshotDB; // position snapshot (NON_PIXEL_SHADER_RESOURCE home state)
	DoubleBufferGpuBuffer::P densitySnapshotDB;  // density snapshot
	DoubleBufferGpuBuffer::P lodSnapshotDB;       // LOD snapshot

	// Liquid surface rendering: single-buffered Texture3D density volume.
	// R32_TYPELESS resource; R32_UINT UAV for CAS float atomic add by splatDensityVolumeCS,
	// R32_FLOAT SRV for liquidPS. Home state: COMMON.
	GpuTexture::P  densityVolume; // VOL_DIM^3, R32_TYPELESS
	CD3DX12_GPU_DESCRIPTOR_HANDLE densityVolumeHandle;     // GPU handle for R32_UINT UAV (used for ClearUnorderedAccessViewUint)
	CD3DX12_CPU_DESCRIPTOR_HANDLE densityVolClearCpuHandle; // CPU handle in static heap for ClearUnorderedAccessViewUint
	UINT liquidTableStartSlot = 0; // first of the 4 contiguous liquidPS SRV slots
	UINT particleSrvTableStart = 0; // first of 3 contiguous SRVs: pos(t0),den(t1),lod(t2)
	UINT cubemapSrvSlot = 0; // slot for the environment cubemap SRV

	// Splat density pipeline: one per-particle CS writes Poly6 contributions via CAS float atomic add.
	ComputeShader::P splatDensityShader; // per-particle: Poly6 splat into densityVolume (R32_UINT UAV)

	// Grid snapshot double buffers: copies of cellCountBuffer and cellPrefixSumBuffer for the graphics queue.
	// Compute writes via CopyBufferRegion in WriteSnapshot(); graphics reads in densityVolumeCS.
	// Home state: NON_PIXEL_SHADER_RESOURCE.
	DoubleBufferGpuBuffer::P cellCountSnapshotDB;
	DoubleBufferGpuBuffer::P cellPrefixSumSnapshotDB;

	Egg::Mesh::Shaded::P liquidMesh;  // fullscreen quad rendered with liquidVS + liquidPS
	float liquidIsoThreshold = 0.5f * RHO0; // density cutoff for liquid/air boundary, tunable via ImGui

	int shadingMode = SHADING_LIQUID; // current particle shading mode, driven by ImGui

	// Create all three descriptor heaps. Must be called before any Init function
	// that populates descriptors.
	void InitDescriptorHeaps();

	void InitConstantBuffers();

	void InitCamera();

	// particleFieldDB[]: one DoubleBufferGpuBuffer per particle attribute.
	// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
	void InitParticleFields();

	// Readback buffers: CPU-readable copies of density and LOD, copied by the compute queue.
	// These are the way we get data back to the CPU. Currently used to display average density
	// and LOD in the UI.
	void InitReadbackBuffers();

	// Double-buffered snapshot buffers for position, density, and LOD (+ grid snapshots).
	void InitSnapshotBuffers();

	// Create the single-buffered density volume texture and its descriptors.
	// The volume is a VOL_DIM^3 R32_TYPELESS Texture3D.
	// R32_UINT UAV (slot DENSITY_VOL_UAV): splatDensityVolumeCS writes float bits via CAS atomic add.
	// R32_FLOAT SRV (slot DENSITY_VOL_SRV): liquidPS reads the same memory as float density.
	// CPU UAV in static heap (slot DENSITY_VOL_UAV_CLEAR): required by ClearUnorderedAccessViewUint.
	// Home state: COMMON. Committed DEFAULT resource -> zero-initialized by OS (= 0.0f for first frame).
	void InitDensityVolume();

	// Load the cubemap texture create GPU resources and descriptors.
	// No GPU commands are recorded here - uploads happen later in UploadAll().
	void InitBackground();

	// Load the solid obstacle; create GPU resources and descriptors.
	// No GPU commands are recorded here - uploads happen later in UploadAll().
	void InitObstacles();

	// Batch all initial data uploads into a single command list execution.
	// All operate on independent resources so there are no state conflicts.
	// Most functions here will be recording GPU commands along the lines of:
	// transition to dest -> copy buffer region ->transition to common/initial state
	void UploadAll();

	// Copy initial particle positions into both snapshot slots so particles are visible
	// before physics starts. Expects command list to be recording.
	void RecordSnapshotUpload();

	// Sets frameCount = 1 and signals computeFence to 1 so the
	// graphics queue's GPU-side wait is immediately satisfied on the first frame.
	//
	// Only computeFence is pre-seeded here. graphicsFence is intentionally NOT pre-seeded:
	// Render() signals graphicsFence to (frameCount - 1) at the end of every frame.
	// On the first Render() call frameCount becomes 2, so the signal targets value 1.
	// If WaitFirstFrame also signaled graphicsFence to 1, the second signal (to the same
	// value) would be rejected by D3D12 (signal value must strictly increase), causing
	// cpuWaitForGraphics(1) to return immediately without waiting for the GPU to finish
	// frame 2's command list. The frame-3 allocator reset and CopyDescriptorsSimple calls
	// would then race against the still-executing frame-2 graphics work, producing:
	//   COMMAND_ALLOCATOR_SYNC (#552) -- allocator reset before GPU finishes
	//   STATIC_DESCRIPTOR_INVALID_DESCRIPTOR_CHANGE (#1001) -- descriptor modified while bound
	void WaitFirstFrame();

	// Build all graphics rendering pipelines (background, particles, DTVS depth-only, liquid, solid transform).
	void BuildGraphicsPipelines();

	// Build the background skybox rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildBackgroundPipeline();

	// Build the particle rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildParticlePipeline();

	// Build the liquid surface rendering pipeline (liquidVS + liquidPS, fullscreen quad).
	// The PS ray-marches through the density volume (t0 = densityVol SRV) and writes SV_Depth
	// for correct depth-buffer occlusion against the solid obstacle.
	void BuildLiquidPipeline();

	// Rebuild world transforms for all obstacles from their obstacleTransforms entries.
	void SetObstacleTransforms();

	ParticleInitData GenerateParticles();

	// Map the upload buffers and copy initial particle data (positions + velocities) from CPU memory
	// to the upload buffers (upload heap, as opposed to default heap).
	// This is a CPU-side operation; the actual GPU transfer is recorded by RecordParticleUpload().
	void FillUploadBuffers(const ParticleInitData& initData);

	// Record copy commands for particle data into the already-open command list.
	// The command list must have been Reset() before calling this.
	void RecordParticleUpload();

	// Create all compute shader PSOs and their descriptor table bindings.
	// Each shader gets its own contiguous region in the main heap; fixedHandle is baked in at creation.
	void BuildComputePipelines();

	void PrepareComputeCommandList();

	void PrepareCommandList();

	void ExecuteGraphics();

	void ExecuteCompute();

	// Override Render() to decouple physics (compute queue) from graphics (direct queue).
	//
	// Physics step: CPU waits for compute frame N-1 to finish (so the allocator can be reused),
	// records compute frame N onto computeList, submits it, signals computeFence to N.
	//
	// Graphics step: graphics queue GPU-waits on frame N-1
	// to ensure the snapshot is ready, records and submits the scene draw, presents,
	// signals graphicsFence, and CPU-waits on it.
	virtual void Render() override;

	void flipDoubleBuffers();

	void RecordComputeCommands();

	void WriteSnapshot();

	void RecordGraphicsCommands();

	// Fill the density volume and draw the ray-marched liquid surface, all on the graphics command list.
	// densityVolumeCS is dispatched here (graphics queue) reading from the previous frame's position and
	// grid snapshots. liquidPS then ray-marches through the freshly filled volume in the same frame.
	void DrawLiquidSurface();

	void BuildImGui();

	void UpdateExternalForce();

	void UpdatePerFrameCb();

	void UpdateComputeCb(float dt);

	virtual void Update(float dt, float T) override;

	void CalculateAvgDensity();

	void CalculateAvgLod();

	// This function cannot be called more than once every targetPeriod time: rate limit
	// a better way of doing this would be a fixed timestep accumulation, where we decouple
	// physics dt from render dt entirely, accumulate wall-clock time, and step physics at a
	// fixed interval
	void Throttle();

public:
	// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
	virtual void CreateSwapChainResources() override;

	virtual void ReleaseSwapChainResources() override;

	// Drain both GPU queues using correct fence values before resizing.
	// AsyncComputeApp::Resize() uses graphicsFrame/computeFrame (always 0)
	// because PbfApp tracks frames via frameCount instead, making those waits no-ops.
	virtual void Resize(int width, int height) override;

	// Allocate all GPU resources that persist across frames: descriptor heaps,
	// buffers for particles and sorting, textures for the environment and obstacle, etc.
	// After this returns, every ID3D12Resource and descriptor heap slot exists,
    // but no data has been uploaded to the GPU yet.
	virtual void CreateResources() override;

	// upload initial data to the GPU and build rendering/compute pipelines.
	virtual void LoadAssets() override;

	// Call once after CreateResources + LoadAssets, from main.cpp where the HWND is available.
	// Sets up ImGui context and its Win32 + D3D12 backends. At this point the D3D12 device, command queue, and
	//imguiSrvHeap all exist.
	void InitImGui(HWND hwnd);

	void ShutdownImGui();

	// Forward window messages (keyboard, mouse) to the camera, and handle app-level hotkeys
	virtual void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override;

	void RunNTimes(int n, bool physicsEnabled);
};
