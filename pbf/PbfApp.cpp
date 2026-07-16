#include "PbfApp.h"
#include <wincodec.h>
#include <filesystem>
#pragma comment(lib, "windowscodecs.lib")

using namespace Egg::Math;

// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
void PbfApp::CreateSwapChainResources() {
	AsyncComputeApp::CreateSwapChainResources();
	// lod and neuralPass are null on the first call (before CreateResources); skip until they exist.
	if (lod)       lod->Resize((UINT)scissorRect.right, (UINT)scissorRect.bottom);
	if (neuralPass) neuralPass->Resize(device.Get(), (UINT)scissorRect.right, (UINT)scissorRect.bottom);
}

// Allocate all GPU resources that persist across frames: descriptor heaps, 
// buffers for particles and sorting, textures for the environment and obstacle, etc.
// After this returns, every ID3D12Resource and descriptor heap slot exists,
// but no data has been uploaded to the GPU yet.
void PbfApp::CreateResources() {
	AsyncComputeApp::CreateResources(); // command allocators, command lists, PSO manager, fences for both queues

	// Heaps must be first: all Init functions below write descriptors into them.
	InitDescriptorHeaps();
	InitConstantBuffers();
	InitCamera();
	InitParticleFields();
	// Create the LOD subsystem now that both descriptor heaps exist. scissorRect was already
	// set by the earlier CreateSwapChainResources call, so width/height are correct.
	lod = LodSystem::Create(device.Get(), numParticles,
		(UINT)scissorRect.right, (UINT)scissorRect.bottom,
		*mainAllocator, *staticAllocator);
	InitReadbackBuffers();
	InitSnapshotBuffers();
	InitDensityVolume();
	InitBackground();
	InitObstacles();
	InitSoftBodyFields();
	InitSoftBodySnapshotBuffers();
	InitSbdGrid();
}

// Create all descriptor allocators. Must be called before any Init function
// that populates descriptors.
void PbfApp::InitDescriptorHeaps() {
	// ImGui SRV heap: 1 slot, shader-visible, exclusively for ImGui's font texture atlas.
	imguiAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, /*shaderVisible*/true);

	// Main shader-visible heap: 256 slots for all per-shader contiguous regions + graphics tables.
	mainAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256, /*shaderVisible*/true);

	// CPU-only static heap: UAV/SRV descriptors written once; source for all CopyDescriptorsSimple calls.
	staticAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 128, /*shaderVisible*/false);
}

void PbfApp::InitConstantBuffers() {
	perFrameCb.CreateResources(device.Get());
	computeCb.CreateResources(device.Get());
}

void PbfApp::InitCamera() {
	camera = Egg::Cam::FirstPerson::Create();
	camera->SetView(Float3(0.0f, 10.0f, -40.0f), Float3(0.0f, 0.0f, 1.0f)); // starting position
	camera->SetSpeed(20.0f);
	camera->SetAspect(aspectRatio);
}

// particleFieldDB[]: double-buffered UAV buffers for each particle attribute.
// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
void PbfApp::InitParticleFields() {
	for (UINT f = 0; f < PF_COUNT; f++) {
		particleFieldDB[f] = DoubleBufferGpuBuffer::Create(
			device.Get(), numParticles, fieldStrides[f],
			(std::wstring(fieldNames[f]) + L" Buffer").c_str(),
			(std::wstring(L"Sorted ") + fieldNames[f] + L" Buffer").c_str(),
			D3D12_RESOURCE_STATE_COMMON,
			*staticAllocator);
	}

	// Upload buffers: CPU-writable staging used once at startup.
	positionUploadBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Position Upload Buffer",
		D3D12_RESOURCE_STATE_GENERIC_READ,
		D3D12_HEAP_TYPE_UPLOAD);
	velocityUploadBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Velocity Upload Buffer",
		D3D12_RESOURCE_STATE_GENERIC_READ,
		D3D12_HEAP_TYPE_UPLOAD);
}

// Readback buffers: CPU-readable copies of density and LOD.
void PbfApp::InitReadbackBuffers() {
	densityReadbackBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(float),
		L"Density Readback Buffer",
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_HEAP_TYPE_READBACK);
	densityReadbackData.resize(numParticles);

	lodReadbackBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(uint32_t),
		L"LOD Readback Buffer",
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_HEAP_TYPE_READBACK);
	lodReadbackData.resize(numParticles);
}

// Double-buffered snapshot buffers for position, density, LOD, cellCount, and cellPrefixSum.
// Snapshot DBs register front SRV targets into the main heap so flip() updates them automatically.
void PbfApp::InitSnapshotBuffers() {
	positionSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Snapshot Position [front]", L"Snapshot Position [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	densitySnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(float),
		L"Snapshot Density [front]", L"Snapshot Density [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	lodSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(UINT),
		L"Snapshot LOD [front]", L"Snapshot LOD [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	// Reserve 3 contiguous main-heap slots for the particle SRV table (t0=pos, t1=den, t2=lod).
	// Snapshot DBs register their front SRV targets here; that's what these 3 srvs are: they are
	// the targets for wiring the snapshot double buffers, in order for the shaders to read 
	// correctly double buffered particle data.
	particleSrvTableStart = mainAllocator->Allocate(3);
	// Register front SRV targets for the particle graphics SRV table (t0=pos, t1=den, t2=lod).
	// flip() keeps these slots pointing at the current front automatically.
	positionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart), true);
	densitySnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart + 1), true);
	lodSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart + 2), true);

	// Grid snapshot DBs (used by densityVolumeCS / splatDensityVolumeCS in liquid mode).
	cellCountSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Count Snapshot [front]", L"Cell Count Snapshot [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	cellPrefixSumSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Prefix Sum Snapshot [front]", L"Cell Prefix Sum Snapshot [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	// 4-slot contiguous liquid table: [density(t0=+0), pos(+1), gridCount(+2), gridPrefix(+3)].
	// density SRV (+0) is static. Snapshot DB front targets handle the other three automatically.
	// we allocate the descriptor space here, and fill slots 1, 2 and 3, and fill slot 0 later
	// in InitDensityVolume
	liquidTableStartSlot = mainAllocator->Allocate(4);
	// Register snapshot front targets: flip() will keep +1/+2/+3 pointing at the current front.
	positionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 1), true);
	cellCountSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 2), true);
	cellPrefixSumSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 3), true);

}

// Single-buffered density volume: VOL_DIM^3, R32_TYPELESS.
// The resource is R32_TYPELESS so that the same 32 bits per voxel can be viewed two ways:
//   - R32_UINT UAV: splatDensityVolumeCS writes float bits via CAS atomic add (InterlockedCompareExchange).
//     CAS operates on integers, so the UAV must be typed as uint even though the data is semantically float.
//   - R32_FLOAT SRV: liquidPS reads those same bits back as a plain float density value.
// The UAV exists in two heaps because ClearUnorderedAccessViewUint has a split CPU/GPU handle contract:
//   - CPU handle: must come from a non-shader-visible (CPU-only) heap so D3D12 can read it on the CPU.
//   - GPU handle: must come from the currently-bound shader-visible heap so the GPU can locate it.
// So we create the canonical UAV in the static (CPU-only) heap and mirror it into the main (GPU-visible) heap.
void PbfApp::InitDensityVolume() {
	const UINT volDim = (UINT)VOL_DIM;
	densityVolume = GpuTexture::Create3D(
		device.Get(), volDim, volDim, volDim,
		DXGI_FORMAT_R32_TYPELESS,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		L"Density Volume",
		D3D12_RESOURCE_STATE_COMMON);

	// Canonical R32_UINT UAV in the static (non-shader-visible) heap.
	// Stored on the texture so GetUavCpuHandle() can be used as:
	//   (a) a CopyDescriptorsSimple source when wiring splatDensityShader in BuildComputePipelines, and
	//   (b) the required CPU handle argument to ClearUnorderedAccessViewUint (saved below).
	densityVolume->CreateUav(device.Get(), *staticAllocator, DXGI_FORMAT_R32_UINT);
	densityVolClearCpuHandle = densityVolume->GetUavCpuHandle();

	// Mirror the UAV into a shader-visible main heap slot.
	// ClearUnorderedAccessViewUint also requires a GPU handle from the currently-bound shader-visible heap.
	// CopyDescriptorsSimple copies the static-heap UAV descriptor into a main-heap slot so we can
	// satisfy that requirement; densityVolumeHandle holds that GPU handle for use in DrawLiquidSurface().
	UINT clearGpuSlot = mainAllocator->Allocate();
	device->CopyDescriptorsSimple(1,
		mainAllocator->GetCpuHandle(clearGpuSlot),
		densityVolClearCpuHandle,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	densityVolumeHandle = mainAllocator->GetGpuHandle(clearGpuSlot);

	// The liquid descriptor table has 4 contiguous slots allocated in InitSnapshotBuffers():
	//   [+0] density volume SRV  <- filled here (static, never changes)
	//   [+1] position snapshot   <- wired via registerFrontTarget in InitSnapshotBuffers()
	//   [+2] cell count snapshot <- wired via registerFrontTarget in InitSnapshotBuffers()
	//   [+3] cell prefix sum     <- wired via registerFrontTarget in InitSnapshotBuffers()
	// We place the SRV directly into the main heap at slot +0 using CreateSrvAt (which targets a
	// specific cpu/gpu handle pair rather than allocating from an allocator). CreateSrvAt does not
	// store the handles on the texture by design, so we call SetSrv to register this as the
	// canonical SRV, keeping the texture in a consistent state.
	densityVolume->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(liquidTableStartSlot),
		mainAllocator->GetGpuHandle(liquidTableStartSlot),
		DXGI_FORMAT_R32_FLOAT);
	densityVolume->SetSrv(
		mainAllocator->GetCpuHandle(liquidTableStartSlot),
		mainAllocator->GetGpuHandle(liquidTableStartSlot));
}

// Load the cubemap texture; create GPU resources and descriptors.
// No GPU commands are recorded here - uploads happen later in UploadAll().
void PbfApp::InitBackground() {
	envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/pretty_sky.dds");

	// SRV at the cubemap slot (t0): sampled by the background pixel shader.
	cubemapSrvSlot = mainAllocator->Allocate();
	envTexture.CreateSRV(device.Get(), mainAllocator->GetHeap(), cubemapSrvSlot);
}

// Load the solid obstacles; create GPU resources and descriptors.
// No GPU commands are recorded here - uploads happen later in UploadAll().
void PbfApp::InitObstacles() {
	// Each entry: obstacle name (= filename stem), initial position, XYZ Euler degrees, uniform scale.
	// To add a new obstacle: bump NUM_OBSTACLES in SharedConfig.hlsli, add a row here.
	struct ObstacleDesc {
		const char* name;
		Float3 position;
		Float3 eulerDeg;
		float  scale;
	};
	static const ObstacleDesc descs[NUM_OBSTACLES] = {
	//	{ "dragonite", Float3(2.3f, -31.0f, 3.5f), Float3(0.0f, 0.0f, 0.0f), 3.8f },
		{ "funnel", Float3(0.0f, 14.0f, 0.0f), Float3(0.0f,  0.0f, 0.0f), 8.0f },
	};

	for (int i = 0; i < NUM_OBSTACLES; i++) {
		obstacles[i] = SolidObstacle::Create();
		obstacles[i]->Load(device.Get(), psoManager, descs[i].name, perFrameCb);
		obstacles[i]->CreateSdfSrv(device.Get(), *staticAllocator);
		obstacles[i]->position = descs[i].position;
		obstacles[i]->eulerDeg = descs[i].eulerDeg;
		obstacles[i]->scale    = descs[i].scale;
	}
}

// Allocate GPU buffers for all SBD fields (position, velocity, predicted position)
// and auxiliary buffers used by the rendering pipeline.
void PbfApp::InitSoftBodyFields() {
	for (UINT f = 0; f < SBD_COUNT; f++) {
		sbdFieldBuffers[f] = GpuBuffer::Create(
			device.Get(), numSbdNodes, sbdFieldStrides[f],
			sbdFieldNames[f], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
		sbdFieldBuffers[f]->CreateUav(device.Get(), *staticAllocator);
	}

	sbdPositionUploadBuffer = GpuBuffer::Create(
		device.Get(), numSbdNodes, sizeof(Float3),
		L"Sbd Position Upload Buffer",
		D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);

	sbdVelocityUploadBuffer = GpuBuffer::Create(
		device.Get(), numSbdNodes, sizeof(Float3),
		L"Sbd Velocity Upload Buffer (zeros)",
		D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
	{
		void* data;
		CD3DX12_RANGE readRange(0, 0);
		DX_API("Failed to map SBD velocity upload buffer")
			sbdVelocityUploadBuffer->Get()->Map(0, &readRange, &data);
		memset(data, 0, (size_t)numSbdNodes * sizeof(Float3));
		sbdVelocityUploadBuffer->Get()->Unmap(0, nullptr);
	}

	// Dummy zero buffers: particleVS needs density (t1) and LOD (t2) even for SBD nodes.
	// DEFAULT heap ensures zero-initialization; we only create SRVs, never UAVs.
	sbdDummyDensityBuffer = GpuBuffer::Create(
		device.Get(), numSbdNodes, sizeof(float),
		L"Sbd Dummy Density", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdDummyDensityBuffer->CreateSrv(device.Get(), *staticAllocator);

	sbdDummyLodBuffer = GpuBuffer::Create(
		device.Get(), numSbdNodes, sizeof(UINT),
		L"Sbd Dummy LOD", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdDummyLodBuffer->CreateSrv(device.Get(), *staticAllocator);
}

// Double-buffered snapshot of SBD node positions for safe graphics readback.
// Compute writes to back; flip() promotes to front for the VS to read.
void PbfApp::InitSoftBodySnapshotBuffers() {
	sbdPositionSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numSbdNodes, sizeof(Float3),
		L"Sbd Position Snapshot [front]", L"Sbd Position Snapshot [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator);

	// Reserve 3 contiguous main-heap SRV slots for the SBD rendering descriptor table.
	// Slot +0: sbdPositionSnapshotDB front (registered below; updated on every flip).
	// Slot +1: sbdDummyDensityBuffer SRV (static, filled in BuildSoftBodyRenderPipeline).
	// Slot +2: sbdDummyLodBuffer SRV     (static, filled in BuildSoftBodyRenderPipeline).
	sbdSrvTableStart = mainAllocator->Allocate(3);
	sbdPositionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(sbdSrvTableStart), true);
}

// Allocate GPU buffers for the SBD spatial grid (cell count, prefix sum, node list,
// and intermediate prefix-sum group buffers). Reuses the same GRID_DIM^3 cell layout as the PBF grid.
void PbfApp::InitSbdGrid() {
	const UINT numCells = GRID_DIM * GRID_DIM * GRID_DIM;
	static constexpr UINT EPG = THREAD_GROUP_SIZE * 2; // elements per prefix-sum group
	const UINT numPass1Groups = numCells / EPG;        // N
	const UINT numPass2Groups = numPass1Groups / EPG;  // M

	sbdCellCountBuffer = GpuBuffer::Create(device.Get(), numCells, sizeof(UINT),
		L"SBD Cell Count", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdCellCountBuffer->CreateUav(device.Get(), *staticAllocator);

	sbdCellPrefixSumBuffer = GpuBuffer::Create(device.Get(), numCells, sizeof(UINT),
		L"SBD Cell Prefix Sum", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdCellPrefixSumBuffer->CreateUav(device.Get(), *staticAllocator);

	sbdNodeListBuffer = GpuBuffer::Create(device.Get(), numSbdNodes, sizeof(UINT),
		L"SBD Node List", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdNodeListBuffer->CreateUav(device.Get(), *staticAllocator);

	sbdGroupSumBuffer = GpuBuffer::Create(device.Get(), numPass1Groups, sizeof(UINT),
		L"SBD Group Sum", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdGroupSumBuffer->CreateUav(device.Get(), *staticAllocator);

	sbdGroupPrefixSumBuffer = GpuBuffer::Create(device.Get(), numPass1Groups, sizeof(UINT),
		L"SBD Group Prefix Sum", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdGroupPrefixSumBuffer->CreateUav(device.Get(), *staticAllocator);

	const UINT superGroupElems = std::max(numPass2Groups, 2u);
	sbdSuperGroupSumBuffer = GpuBuffer::Create(device.Get(), superGroupElems, sizeof(UINT),
		L"SBD Super Group Sum", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	sbdSuperGroupSumBuffer->CreateUav(device.Get(), *staticAllocator);
}

// Fill the CPU-side BCC position upload buffer.
// Sublattice A: SBD_DIM_X * SBD_DIM_Y * SBD_DIM_Z nodes at (i, j, k) * spacing.
// Sublattice B: (SBD_DIM_X+1)*(SBD_DIM_Y+1)*(SBD_DIM_Z+1) nodes at (i-0.5, j-0.5, k-0.5)*spacing,
//              so B extends half a spacing unit outside A on every side.
// Both sublattices share the same world-space center.
void PbfApp::FillSbdUploadBuffer() {
	const float spacing = 2.0f;
	const Float3 center(0.0f, 0.0f, 0.0f);
	// A center = (DIM-1)*spacing/2 per axis; offset places A node (0,0,0) at center - that.
	const Float3 aOffset = center - Float3(SBD_DIM_X - 1, SBD_DIM_Y - 1, SBD_DIM_Z - 1) * (spacing * 0.5f);
	// B starts half a spacing below A, so B node (0,0,0) = aOffset - (0.5,0.5,0.5)*spacing.
	const Float3 bOffset = aOffset - Float3(0.5f, 0.5f, 0.5f) * spacing;

	std::vector<Float3> positions;
	positions.reserve(numSbdNodes);

	// Sublattice A: SBD_DIM_X * SBD_DIM_Y * SBD_DIM_Z nodes
	for (int z = 0; z < SBD_DIM_Z; z++)
		for (int y = 0; y < SBD_DIM_Y; y++)
			for (int x = 0; x < SBD_DIM_X; x++)
				positions.push_back(aOffset + Float3::Random() * 5.5f + Float3((float)x, (float)y, (float)z) * spacing);

	// Sublattice B: (SBD_DIM_X+1) * (SBD_DIM_Y+1) * (SBD_DIM_Z+1) nodes
	for (int z = 0; z <= SBD_DIM_Z; z++)
		for (int y = 0; y <= SBD_DIM_Y; y++)
			for (int x = 0; x <= SBD_DIM_X; x++)
				positions.push_back(bOffset + Float3::Random() * 5.5f + Float3((float)x, (float)y, (float)z) * spacing);

	void* data;
	CD3DX12_RANGE readRange(0, 0);
	DX_API("Failed to map SBD position upload buffer")
		sbdPositionUploadBuffer->Get()->Map(0, &readRange, &data);
	memcpy(data, positions.data(), positions.size() * sizeof(Float3));
	sbdPositionUploadBuffer->Get()->Unmap(0, nullptr);
}

// Record GPU upload commands: copy initial BCC positions into the main buffer and both
// snapshot slots, then transition all SBD buffers to their permanent home states.
void PbfApp::RecordSbdUpload() {
	const UINT64 posBytes = (UINT64)numSbdNodes * sizeof(Float3);

	// Upload positions: upload buffer → SBD position field buffer
	sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	commandList->CopyBufferRegion(sbdFieldBuffers[SBD_POSITION]->Get(), 0,
		sbdPositionUploadBuffer->Get(), 0, posBytes);

	// Copy to both snapshot slots for initial visibility before the first physics step.
	sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, commandList.Get());
	sbdPositionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	sbdPositionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	commandList->CopyBufferRegion(sbdPositionSnapshotDB->getFront()->Get(), 0,
		sbdFieldBuffers[SBD_POSITION]->Get(), 0, posBytes);
	commandList->CopyBufferRegion(sbdPositionSnapshotDB->getBack()->Get(), 0,
		sbdFieldBuffers[SBD_POSITION]->Get(), 0, posBytes);

	// Transition all SBD buffers to their permanent home states.
	sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	sbdFieldBuffers[SBD_VELOCITY]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	sbdFieldBuffers[SBD_PREDICTED_POSITION]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	sbdPositionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	sbdPositionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());

	// Dummy SRV buffers never change state again after this.
	sbdDummyDensityBuffer->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	sbdDummyLodBuffer->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
}

// Wire the SBD compute shader descriptor tables and build their PSOs.
void PbfApp::BuildSoftBodyComputePipelines() {
	D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();
	using P = com_ptr<ID3D12Resource>*;

	auto copyToMain = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
		device->CopyDescriptorsSimple(1, mainAllocator->GetCpuHandle(slot), src,
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	};

	// sbdPredictCS: UAV(u0-2) = 3 slots (position, velocity, predictedPosition)
	{
		UINT s = mainAllocator->Allocate(3);
		copyToMain(s,     sbdFieldBuffers[SBD_POSITION]->GetUavCpuHandle());
		copyToMain(s + 1, sbdFieldBuffers[SBD_VELOCITY]->GetUavCpuHandle());
		copyToMain(s + 2, sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetUavCpuHandle());
		sbdPredictShader = ComputeShader::Create(device.Get(), "Shaders/sbdPredictCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdFieldBuffers[SBD_POSITION]->GetResourcePtr(),
			                sbdFieldBuffers[SBD_VELOCITY]->GetResourcePtr() },
			std::vector<P>{ sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetResourcePtr() });
	}

	// sbdStrainCS: UAV(u0) = 1 slot (predictedPosition)
	{
		UINT s = mainAllocator->Allocate(1);
		copyToMain(s, sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetUavCpuHandle());
		sbdStrainShader = ComputeShader::Create(device.Get(), "Shaders/sbdStrainCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{},
			std::vector<P>{ sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetResourcePtr() });
	}

	// sbdUpdateVelocityCS: UAV(u0-2) = 3 slots (position, velocity, predictedPosition)
	{
		UINT s = mainAllocator->Allocate(3);
		copyToMain(s,     sbdFieldBuffers[SBD_POSITION]->GetUavCpuHandle());
		copyToMain(s + 1, sbdFieldBuffers[SBD_VELOCITY]->GetUavCpuHandle());
		copyToMain(s + 2, sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetUavCpuHandle());
		sbdUpdateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/sbdUpdateVelocityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdFieldBuffers[SBD_POSITION]->GetResourcePtr(),
			                sbdFieldBuffers[SBD_PREDICTED_POSITION]->GetResourcePtr() },
			std::vector<P>{ sbdFieldBuffers[SBD_VELOCITY]->GetResourcePtr(),
			                sbdFieldBuffers[SBD_POSITION]->GetResourcePtr() });
	}

	BuildSbdGridPipelines();
}

// Wire descriptor tables for the SBD spatial-grid shaders and build their PSOs.
// Reuses existing prefix-sum CSOs (generic buffer operations, grid-size agnostic).
void PbfApp::BuildSbdGridPipelines() {
	D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();
	using P = com_ptr<ID3D12Resource>*;
	auto copyToMain = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
		device->CopyDescriptorsSimple(1, mainAllocator->GetCpuHandle(slot), src,
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	};

	// sbdClearGridShader: reuses clearGridCS.cso — UAV(u0) = sbdCellCount
	{
		UINT s = mainAllocator->Allocate(1);
		copyToMain(s, sbdCellCountBuffer->GetUavCpuHandle());
		sbdClearGridShader = ComputeShader::Create(device.Get(), "Shaders/clearGridCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdCellCountBuffer->GetResourcePtr() },
			std::vector<P>{ sbdCellCountBuffer->GetResourcePtr() });
	}

	// sbdCountGridShader: UAV(u0-1) = [sbdPosition, sbdCellCount]
	{
		UINT s = mainAllocator->Allocate(2);
		copyToMain(s,     sbdFieldBuffers[SBD_POSITION]->GetUavCpuHandle());
		copyToMain(s + 1, sbdCellCountBuffer->GetUavCpuHandle());
		sbdCountGridShader = ComputeShader::Create(device.Get(), "Shaders/sbdCountGridCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdFieldBuffers[SBD_POSITION]->GetResourcePtr() },
			std::vector<P>{ sbdCellCountBuffer->GetResourcePtr() });
	}

	// Prefix-sum passes 1–5: reuse PBF prefix-sum CSOs, new SBD buffer bindings.
	// Pass 1: [sbdCellCount, sbdCellPrefixSum, sbdGroupSum]
	{
		UINT s = mainAllocator->Allocate(3);
		copyToMain(s,     sbdCellCountBuffer->GetUavCpuHandle());
		copyToMain(s + 1, sbdCellPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 2, sbdGroupSumBuffer->GetUavCpuHandle());
		sbdGridPass1 = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass1_2CS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdCellCountBuffer->GetResourcePtr() },
			std::vector<P>{ sbdCellPrefixSumBuffer->GetResourcePtr(), sbdGroupSumBuffer->GetResourcePtr() });
	}
	// Pass 2: [sbdGroupSum, sbdGroupPrefixSum, sbdSuperGroupSum]
	{
		UINT s = mainAllocator->Allocate(3);
		copyToMain(s,     sbdGroupSumBuffer->GetUavCpuHandle());
		copyToMain(s + 1, sbdGroupPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 2, sbdSuperGroupSumBuffer->GetUavCpuHandle());
		sbdGridPass2 = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass1_2CS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdGroupSumBuffer->GetResourcePtr() },
			std::vector<P>{ sbdGroupPrefixSumBuffer->GetResourcePtr(), sbdSuperGroupSumBuffer->GetResourcePtr() });
	}
	// Pass 3: [sbdSuperGroupSum] in-place
	{
		UINT s = mainAllocator->Allocate(1);
		copyToMain(s, sbdSuperGroupSumBuffer->GetUavCpuHandle());
		sbdGridPass3 = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass3CS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdSuperGroupSumBuffer->GetResourcePtr() },
			std::vector<P>{ sbdSuperGroupSumBuffer->GetResourcePtr() });
	}
	// Pass 4: [sbdGroupPrefixSum, sbdSuperGroupSum]
	{
		UINT s = mainAllocator->Allocate(2);
		copyToMain(s,     sbdGroupPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 1, sbdSuperGroupSumBuffer->GetUavCpuHandle());
		sbdGridPass4 = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass4_5CS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdSuperGroupSumBuffer->GetResourcePtr(), sbdGroupPrefixSumBuffer->GetResourcePtr() },
			std::vector<P>{ sbdGroupPrefixSumBuffer->GetResourcePtr() });
	}
	// Pass 5: [sbdCellPrefixSum, sbdGroupPrefixSum]
	{
		UINT s = mainAllocator->Allocate(2);
		copyToMain(s,     sbdCellPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 1, sbdGroupPrefixSumBuffer->GetUavCpuHandle());
		sbdGridPass5 = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass4_5CS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdGroupPrefixSumBuffer->GetResourcePtr(), sbdCellPrefixSumBuffer->GetResourcePtr() },
			std::vector<P>{ sbdCellPrefixSumBuffer->GetResourcePtr() });
	}

	// sbdSortShader: UAV(u0-3) = [sbdPosition, sbdCellCount, sbdCellPrefixSum, sbdNodeList]
	{
		UINT s = mainAllocator->Allocate(4);
		copyToMain(s,     sbdFieldBuffers[SBD_POSITION]->GetUavCpuHandle());
		copyToMain(s + 1, sbdCellCountBuffer->GetUavCpuHandle());
		copyToMain(s + 2, sbdCellPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 3, sbdNodeListBuffer->GetUavCpuHandle());
		sbdSortShader = ComputeShader::Create(device.Get(), "Shaders/sbdSortCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ sbdFieldBuffers[SBD_POSITION]->GetResourcePtr(),
			                sbdCellPrefixSumBuffer->GetResourcePtr(),
			                sbdCellCountBuffer->GetResourcePtr() },
			std::vector<P>{ sbdNodeListBuffer->GetResourcePtr(),
			                sbdCellCountBuffer->GetResourcePtr() });
	}

	// sbdPoreSuctionShader: UAV(u0-4) =
	//   [fluidPredPos back, sbdPosition, sbdCellCount, sbdCellPrefixSum, sbdNodeList]
	{
		UINT s = mainAllocator->Allocate(5);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copyToMain(s + 1, sbdFieldBuffers[SBD_POSITION]->GetUavCpuHandle());
		copyToMain(s + 2, sbdCellCountBuffer->GetUavCpuHandle());
		copyToMain(s + 3, sbdCellPrefixSumBuffer->GetUavCpuHandle());
		copyToMain(s + 4, sbdNodeListBuffer->GetUavCpuHandle());
		sbdPoreSuctionShader = ComputeShader::Create(device.Get(), "Shaders/sbdPoreSuctionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                sbdFieldBuffers[SBD_POSITION]->GetResourcePtr(),
			                sbdCellCountBuffer->GetResourcePtr(),
			                sbdCellPrefixSumBuffer->GetResourcePtr(),
			                sbdNodeListBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() });
	}
}

// Dispatch the SBD spatial-grid build sequence on cmd:
//   clear → count → 5-pass prefix sum → clear → sort
// After this, sbdCellCount[ci] = node count, sbdCellPrefixSum[ci] = cumulative offset,
// sbdNodeList[prefix[ci]..prefix[ci]+count[ci]-1] = SBD node indices in that cell.
void PbfApp::RecordSbdGridBuild(ID3D12GraphicsCommandList* cmd) {
	const UINT numCells     = GRID_DIM * GRID_DIM * GRID_DIM;
	const UINT numCellGroups = (numCells + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
	const UINT EPG           = THREAD_GROUP_SIZE * 2;
	const UINT N             = numCells / EPG;       // pass-1 groups
	const UINT M             = N / EPG;              // pass-2 groups
	const UINT sbdNodeGroups = (SBD_NUM_NODES + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	sbdClearGridShader->dispatch_then_barrier(cmd, numCellGroups);
	sbdCountGridShader->dispatch_then_barrier(cmd, sbdNodeGroups);
	sbdGridPass1->dispatch_then_barrier(cmd, N);
	sbdGridPass2->dispatch_then_barrier(cmd, M);
	sbdGridPass3->dispatch_then_barrier(cmd, 1);
	sbdGridPass4->dispatch_then_barrier(cmd, M);
	sbdGridPass5->dispatch_then_barrier(cmd, N);
	sbdClearGridShader->dispatch_then_barrier(cmd, numCellGroups);
	sbdSortShader->dispatch_then_barrier(cmd, sbdNodeGroups);
}

// Build the SBD node rendering pipeline, reusing the particle VS/GS/PS.
// Binds a 3-slot SRV table: sbdPositionSnapshot (t0), dummy density (t1), dummy LOD (t2).
void PbfApp::BuildSoftBodyRenderPipeline() {
	auto copyToMain = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
		device->CopyDescriptorsSimple(1, mainAllocator->GetCpuHandle(slot), src,
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	};

	// Fill the static SRV slots for the dummy buffers (slot 0 was already wired by registerFrontTarget).
	copyToMain(sbdSrvTableStart + 1, sbdDummyDensityBuffer->GetSrvCpuHandle());
	copyToMain(sbdSrvTableStart + 2, sbdDummyLodBuffer->GetSrvCpuHandle());

	// Reuse the same particle shaders; only the SRV table offset and vertex count differ.
	com_ptr<ID3DBlob> vertexShader   = Egg::Shader::LoadCso("Shaders/particleVS.cso");
	com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
	com_ptr<ID3DBlob> pixelShader    = Egg::Shader::LoadCso("Shaders/particlePS.cso");
	com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

	Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
	material->SetRootSignature(rootSig);
	material->SetVertexShader(vertexShader);
	material->SetGeometryShader(geometryShader);
	material->SetPixelShader(pixelShader);
	material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
	material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
	material->SetConstantBuffer(perFrameCb);
	material->SetSrvHeap(1, mainAllocator->GetHeap(), sbdSrvTableStart * mainAllocator->GetDescriptorSize());

	Egg::Mesh::Geometry::P geometry = Egg::Mesh::NullGeometry::Create(numSbdNodes);
	geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

	sbdMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
}

// upload initial data to the GPU and build rendering/compute pipelines.
void PbfApp::LoadAssets() {
	UploadAll();
	BuildGraphicsPipelines();
	BuildComputePipelines();

	// Neural post-process pass. Disabled gracefully if the ONNX file is missing.
	neuralPass = NeuralPostProcess::Create(
		device.Get(), commandQueue.Get(), "models/unet_postprocess.onnx-disabled");
	if (neuralPass && neuralPass->enabled)
		neuralPass->Resize(device.Get(), (UINT)scissorRect.right, (UINT)scissorRect.bottom);
}

// Batch all initial data uploads into a single command list execution.
// All operate on independent resources so there are no state conflicts.
// Most functions here will be recording GPU commands along the lines of:
// transition to dest -> copy buffer region ->transition to common/initial state
void PbfApp::UploadAll() {
	ParticleInitData initData = GenerateParticles(); // initData holds particle data on CPU
	FillUploadBuffers(initData); // memcpy particle data into the upload buffers on the GPU

	DX_API("Failed to reset command allocator (UploadAll)")
		commandAllocator->Reset();
	DX_API("Failed to reset command list (UploadAll)")
		commandList->Reset(commandAllocator.Get(), nullptr);

	envTexture.UploadResource(commandList.Get()); // record cubemap copy + barrier
	RecordParticleUpload(); // record particle copy + barriers
	for (int i = 0; i < NUM_OBSTACLES; i++)
		obstacles[i]->UploadSdf(commandList.Get()); // record SDF texture copy + barrier
	RecordSnapshotUpload(); // record initial state of snapshot buffers for frame 1
	FillSbdUploadBuffer();  // CPU: generate BCC positions into the upload buffer
	RecordSbdUpload();      // GPU: copy BCC positions to field buffer + both snapshot slots

	// Pre-clear both depth texture slots to 1.0 (far plane) so the first DTVS compute frame
	// sees valid depth data even before any graphics depth pass has run.
	lod->RecordDepthClear(commandList.Get());

	DX_API("Failed to close command list (UploadAll)")
		commandList->Close();
	ID3D12CommandList* cls[] = { commandList.Get() };
	commandQueue->ExecuteCommandLists(_countof(cls), cls);

	WaitFirstFrame(); // GPU-wait until the above uploads are finished and the snapshot buffers are ready for use in frame 1

	// the upload heap copies are done - free the temporary upload resources
	envTexture.ReleaseUploadResources();
	for (int i = 0; i < NUM_OBSTACLES; i++)
		obstacles[i]->ReleaseUploadResources();
}

ParticleInitData PbfApp::GenerateParticles() {
	// create and return an evenly spaced grid of particles so we can see something on screen
	ParticleInitData data;
	Float3 grid = Float3(particlesX, particlesY, particlesZ);
	Float3 offset = -(grid * particleSpacing) / 2.0f; // shift so the cube is centered at the origin
	offset += Float3(offsetX, offsetY, offsetZ); // apply user-defined world space offset
	for (int x = 0; x < grid.x; x++) {
		for (int y = 0; y < grid.y; y++) {
			for (int z = 0; z < grid.z; z++) {
				data.positions.push_back(offset + Float3(x, y, z) * particleSpacing);
				data.velocities.push_back(Float3(0.0f, 0.0f, 0.0f)); // start at rest
			}
		}
	}
	return data;
}

// Map the upload buffers and copy initial particle data (positions + velocities) from CPU memory
// to the upload buffers (upload heap, as opposed to default heap).
// This is a CPU-side operation; the actual GPU transfer is recorded by RecordParticleUpload().
void PbfApp::FillUploadBuffers(const ParticleInitData& initData) {
	void* posData; // will point to the mapped CPU memory of the upload buffer
	CD3DX12_RANGE readRange(0, 0); // empty range: CPU won't read anything from this buffer
	// make the upload buffer's memory CPU accessible, i.e. positionUploadBuffer - posData association
	// 0 is subresource index, on success, posData points to the buffer
	DX_API("Failed to map position upload buffer")
		positionUploadBuffer->Get()->Map(0, &readRange, &posData);
	memcpy(posData, initData.positions.data(), initData.positions.size() * sizeof(Float3));
	positionUploadBuffer->Get()->Unmap(0, nullptr);

	void* velData;
	DX_API("Failed to map velocity upload buffer")
		velocityUploadBuffer->Get()->Map(0, &readRange, &velData);
	memcpy(velData, initData.velocities.data(), initData.velocities.size() * sizeof(Float3));
	velocityUploadBuffer->Get()->Unmap(0, nullptr);
}


// Record copy commands for particle data into the already-open command list.
// The command list must have been Reset() before calling this.
void PbfApp::RecordParticleUpload() {
	// Copy to both front and back so that the CPU-side flip() at the start of the first physics
	// frame leaves a valid initial state in whichever buffer becomes the new front.
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	commandList->CopyBufferRegion(particleFieldDB[PF_POSITION]->getFront()->Get(), 0,
		positionUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_VELOCITY]->getFront()->Get(), 0,
		velocityUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_POSITION]->getBack()->Get(), 0,
		positionUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_VELOCITY]->getBack()->Get(), 0,
		velocityUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
}

// Copy initial particle positions into both snapshot slots so particles are visible
// before physics starts. Expects command list to be recording.
void PbfApp::RecordSnapshotUpload() {
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, commandList.Get());
	positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());

	const UINT64 posBytes = (UINT64)numParticles * sizeof(Float3);
	commandList->CopyBufferRegion(positionSnapshotDB->getFront()->Get(), 0,
		particleFieldDB[PF_POSITION]->getFront()->Get(), 0, posBytes);
	commandList->CopyBufferRegion(positionSnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_POSITION]->getFront()->Get(), 0, posBytes);

	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());

	// Transition all remaining snapshot buffers from COMMON to their home NON_PIXEL_SHADER_RESOURCE state.
	densitySnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	lodSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellCountSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
}

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
void PbfApp::WaitFirstFrame() {
	frameCount = 1;
	computeFence.signal(commandQueue, frameCount); // when reached, we're done calculating frame 1
	cpuWaitForCompute(frameCount);
	// graphicsFence deliberately NOT pre-seeded here (see comment above).

	lastFrame = clock::now();
}

// Build all graphics rendering pipelines (background, particles, liquid, solid transform, SBD nodes).
// The DTVS depth-only pipeline is built inside lod->BuildPipelines().
void PbfApp::BuildGraphicsPipelines() {
	BuildBackgroundPipeline();
	BuildParticlePipeline();
	BuildLiquidPipeline();
	SetObstacleTransforms();
	BuildSoftBodyRenderPipeline();
}

// Build the background skybox rendering pipeline (shaders, material, mesh).
// Called after all resources and descriptors are ready.
void PbfApp::BuildBackgroundPipeline() {
	// loadCso reads the pre-compiled .cso binary into a blob
	com_ptr<ID3DBlob> bgVertexShader = Egg::Shader::LoadCso("Shaders/bgVS.cso"); // vertex shader
	com_ptr<ID3DBlob> bgPixelShader = Egg::Shader::LoadCso("Shaders/bgPS.cso"); // pixel shader
	com_ptr<ID3D12RootSignature> bgRootSig = Egg::Shader::LoadRootSignature(device.Get(), bgVertexShader.Get());

	// the material of a mesh is what handles shader configuration, which includes root signature,
	// shader bytecode, pipeline state settings, and resource bindings (SRV/UAV/CBV)
	Egg::Mesh::Material::P bgMaterial = Egg::Mesh::Material::Create();
	bgMaterial->SetRootSignature(bgRootSig);
	bgMaterial->SetVertexShader(bgVertexShader);
	bgMaterial->SetPixelShader(bgPixelShader);
	// enable depth testing - the background writes z=0.999999, so particles (closer) will draw in front
	bgMaterial->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
	bgMaterial->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
	// bind the per-frame constant buffer (root parameter 0)
	bgMaterial->SetConstantBuffer(perFrameCb);
	// bind the SRV heap containing the cubemap (root parameter 1, starting at the cubemap slot)
	bgMaterial->SetSrvHeap(1, mainAllocator->GetHeap(), cubemapSrvSlot * mainAllocator->GetDescriptorSize());

	// The fullscreen quad from Egg's prefab library - 2 triangles covering the entire screen
	// the geometry of a mesh is what handles raw vertex data
	Egg::Mesh::Geometry::P bgGeometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

	// Mesh = material + geometry
	backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, bgMaterial, bgGeometry);
}

// Build the particle rendering pipeline (shaders, material, mesh).
// Called after all resources and descriptors are ready.
void PbfApp::BuildParticlePipeline() {
	// loadCso reads the pre-compiled .cso binary into a blob
	com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/particleVS.cso");
	com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
	com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/particlePS.cso");
	// extract the root signature from the vertex shader
	// the [RootSignature(...)] attribute we defined in the HLSL gets embedded in the compiled blob
	com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

	// create a material to hold shaders, root signature, blend/rasterizer/depth state
	Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
	material->SetRootSignature(rootSig);
	material->SetVertexShader(vertexShader);
	material->SetGeometryShader(geometryShader); // expand points into quads on the GPU
	material->SetPixelShader(pixelShader);
	// enable depth testing so particles occlude each other correctly
	material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
	material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
	// bind the per-frame constant buffer so the shader can access camera matrices
	material->SetConstantBuffer(perFrameCb);
	// bind the particle SRV table (slots 1-3 in srvHeap) to root parameter 1 so the VS
	// can read position (t0), density (t1), and LOD (t2).
	// SetSrvHeap's third argument is a raw byte offset into the heap, not a descriptor slot index,
	// so we must multiply the slot index by the descriptor increment size to get the correct byte offset
	material->SetSrvHeap(1, mainAllocator->GetHeap(), particleSrvTableStart * mainAllocator->GetDescriptorSize());

	// NullGeometry: no vertex buffer - the VS fetches positions from the structured buffer using SV_VertexID
	// numParticles tells DrawInstanced how many vertices (and therefore SV_VertexID values) to generate
	Egg::Mesh::Geometry::P geometry = Egg::Mesh::NullGeometry::Create(numParticles);
	geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST); // each SV_VertexID maps to one point, expanded to a quad by the GS

	// mesh = material + geometry + PSO (created by PSO manager based on the material's root signature, shaders, and states)
	particleMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
}

// Build the liquid surface rendering pipeline (liquidVS + liquidPS, fullscreen quad).
// The PS ray-marches through the density volume (t0 = densityVol SRV) and writes SV_Depth
// for correct depth-buffer occlusion against the solid obstacle.
void PbfApp::BuildLiquidPipeline() {
	com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/liquidVS.cso");
	com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/liquidPS.cso");
	// extract the root signature from the vertex shader blob (same LiquidRootSig as in PS)
	com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

	Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
	material->SetRootSignature(rootSig);
	material->SetVertexShader(vertexShader);
	material->SetPixelShader(pixelShader);
	// depth test + write: liquid SV_Depth correctly occludes / is occluded by the solid obstacle
	material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
	material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
	// per-frame CB (root param 0): camera matrices, bbMin/bbMax, threshold
	material->SetConstantBuffer(perFrameCb);
	// Single 4-slot descriptor table (t0..t3) starting at liquidTableStartSlot.
	// density (static), position (per-frame), grid count (per-frame), grid prefix (per-frame).
	material->SetSrvHeap(1, mainAllocator->GetHeap(), liquidTableStartSlot * mainAllocator->GetDescriptorSize());

	// same fullscreen quad as the background skybox
	Egg::Mesh::Geometry::P geometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

	liquidMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
}

// Rebuild world transforms for all obstacles from each obstacle's own transform fields.
void PbfApp::SetObstacleTransforms() {
	for (int i = 0; i < NUM_OBSTACLES; i++)
		obstacles[i]->RebuildTransform();
}

// Create all compute shader PSOs and wire each shader to its contiguous descriptor region.
// Each shader's region is allocated from mainAllocator; front/back DB targets are registered
// so flip() keeps the descriptors current without any per-frame CopyDescriptorsSimple in hot paths.
// TODO: double check inputs/outputs/bindings for each shader, especially the front vs back buffer targets
void PbfApp::BuildComputePipelines() {
	D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();
	using P = com_ptr<ID3D12Resource>*;

	// lambda to copy a single descriptor from src (a static heap slot) to a main heap slot
	// [&] captures everything by reference, the lambda takes 2 parameters, slot and src, and copies
	// 1 descriptor from src to the main heap slot at index slot.
	auto copyToMain = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
		device->CopyDescriptorsSimple(1, mainAllocator->GetCpuHandle(slot), src,
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	};
	// SpatialGrid owns cellCount/cellPrefixSum/perm/groupSum buffers and all grid+sort shaders.
	// Must be created first: solver shaders below reference the grid buffers via GetCellCountBuffer()/GetPrefixSumBuffer().
	spatialGrid = SpatialGrid::Create(device.Get(), numCells, numParticles,
		*mainAllocator, *staticAllocator, cbv, particleFieldDB);

	// predictCS: UAV(u0-2) SRV(t0..NUM_OBSTACLES-1) = 3 + NUM_OBSTACLES slots
	// [0]=position, [1]=velocity, [2]=predictedPosition, [3..3+NUM_OBSTACLES-1]=SDF SRVs
	{
		UINT s = mainAllocator->Allocate(3 + NUM_OBSTACLES);
		particleFieldDB[PF_POSITION]->registerFrontTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerFrontTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerFrontTarget(mainAllocator->GetCpuHandle(s + 2), false);
		for (int i = 0; i < NUM_OBSTACLES; i++)
			copyToMain(s + 3 + i, obstacles[i]->GetSdfCpuHandle());
		predictShader = ComputeShader::Create(device.Get(), "Shaders/predictCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getFront()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getFront()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getFront()->GetResourcePtr(),
			                particleFieldDB[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr() });
	}

	// collisionPredictedPositionCS: UAV(u0-1) SRV(t0..NUM_OBSTACLES-1) = 2 + NUM_OBSTACLES slots
	// [0]=predictedPosition back, [1]=lod, [2..2+NUM_OBSTACLES-1]=SDF SRVs
	{
		UINT s = mainAllocator->Allocate(2 + NUM_OBSTACLES);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copyToMain(s + 1, lod->GetLodBuffer()->GetUavCpuHandle());
		for (int i = 0; i < NUM_OBSTACLES; i++)
			copyToMain(s + 2 + i, obstacles[i]->GetSdfCpuHandle());
		collisionPredictedPositionShader = ComputeShader::Create(device.Get(), "Shaders/collisionPredictedPositionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() });
	}

	// positionFromScratchCS: UAV(u0-2) = 3 slots 
	// [0]=predictedPosition back, [1]=scratch back, [2]=lod
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		copyToMain(s + 2, lod->GetLodBuffer()->GetUavCpuHandle());
		positionFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/positionFromScratchCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() });
	}

	// updateVelocityCS: UAV(u0-2) = 3 slots 
	// [0]=position back, [1]=velocity back, [2]=predictedPosition back
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		updateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/updateVelocityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr() });
	}

	// velocityFromScratchCS: UAV(u0-1) = 2 slots 
	// [0]=velocity back, [1]=scratch back
	{
		UINT s = mainAllocator->Allocate(2);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		velocityFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/velocityFromScratchCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr() });
	}

	// updatePositionCS: UAV(u0-1) = 2 slots 
	// [0]=position back, [1]=predictedPosition back
	{
		UINT s = mainAllocator->Allocate(2);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),          false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		updatePositionShader = ComputeShader::Create(device.Get(), "Shaders/updatePositionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr() });
	}

	// lambdaCS: UAV(u0-5) = 6 slots 
	// [0]=predictedPosition, [1]=lambda, [2]=density, [3]=cellCount, [4]=cellPrefixSum, [5]=lod
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_DENSITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, lod->GetLodBuffer()->GetUavCpuHandle());
		lambdaShader = ComputeShader::Create(device.Get(), "Shaders/lambdaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_DENSITY]->getBack()->GetResourcePtr() });
	}

	// deltaCS: UAV(u0-5) = 6 slots
	// [0]=predictedPosition, [1]=lambda, [2]=scratch, [3]=cellCount, [4]=cellPrefixSum, [5]=lod
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, lod->GetLodBuffer()->GetUavCpuHandle());
		deltaShader = ComputeShader::Create(device.Get(), "Shaders/deltaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// vorticityCS: UAV(u0-4) = 5 slots 
	// [0]=position, [1]=velocity, [2]=omega, [3]=cellCount, [4]=cellPrefixSum
	{
		UINT s = mainAllocator->Allocate(5);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		vorticityShader = ComputeShader::Create(device.Get(), "Shaders/vorticityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr() });
	}

	// confinementViscosityCS: UAV(u0-5) = 6 slots 
	// [0]=position, [1]=velocity, [2]=omega, [3]=scratch, [4]=cellCount, [5]=cellPrefixSum
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 3), false);
		copyToMain(s + 4, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		confinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/confinementViscosityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// LOD shaders (DTC path, DTVS path, non-adaptive) and the depth-only graphics PSO
	// are all built inside the LOD subsystem.
	lod->BuildPipelines(device.Get(), cbv, *mainAllocator, particleFieldDB);

	BuildSoftBodyComputePipelines();

	// splatDensityVolumeCS: SRV(t0) UAV(u0) = 2 slots 
	// [0]=posSnapshot front SRV, [1]=densityVol UAV
	{
		UINT s = mainAllocator->Allocate(2);
		positionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(s), true);
		copyToMain(s + 1, densityVolume->GetUavCpuHandle());
		splatDensityShader = ComputeShader::Create(device.Get(), "Shaders/splatDensityVolumeCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ positionSnapshotDB->getFront()->GetResourcePtr()},
			std::vector<P>{ densityVolume->GetResourcePtr() });
	}

	// GSM variants 

	// GSM_lambdaCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_DENSITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, lod->GetLodBuffer()->GetUavCpuHandle());
		gsmLambdaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_lambdaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_DENSITY]->getBack()->GetResourcePtr() });
	}

	// GSM_deltaCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, lod->GetLodBuffer()->GetUavCpuHandle());
		gsmDeltaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_deltaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lod->GetLodBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// GSM_vorticity: UAV(u0-4) = 5 slots
	{
		UINT s = mainAllocator->Allocate(5);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copyToMain(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		gsmVorticityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_vorticity.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr() });
	}

	// GSM_confinementViscosityCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 3), false);
		copyToMain(s + 4, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copyToMain(s + 5, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		gsmConfinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_confinementViscosityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}
}

// Call once after CreateResources + LoadAssets, from main.cpp where the HWND is available.
// Sets up ImGui context and its Win32 + D3D12 backends. At this point the D3D12 device, command queue, and
//imguiSrvHeap all exist.
void PbfApp::InitImGui(HWND hwnd) {
	IMGUI_CHECKVERSION(); // checks that the headers and compiled .lib are from the same version of ImGui
	// create the ImGui context, which stores ImGui's internal state and is needed before calling any ImGui functions
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplWin32_Init(hwnd); // Win32 backend: handles mouse position, keyboard input, cursor shape

	// D3D12 backend: renders ImGui's vertex/index buffers using our device and command queue.
	// We use the legacy single-descriptor path: one SRV for the font texture atlas.
	// Internally ImGui_ImplDX12_Init creates a root signature and PSO, allocates
	// a two vertex/index buffers for swapping, creatres its own command allocator and command list,
	// writes the font texture srv into LegacySingleSrvCpuDescriptor and LegacySingleSrvGpuDescriptor
	ImGui_ImplDX12_InitInfo initInfo;
	initInfo.Device = device.Get();
	initInfo.CommandQueue = commandQueue.Get();
	initInfo.NumFramesInFlight = 2; // matches our double-buffered swap chain
	initInfo.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM; // must match swap chain format
	initInfo.SrvDescriptorHeap = imguiAllocator->GetHeap();
	initInfo.LegacySingleSrvCpuDescriptor = imguiAllocator->GetCpuHandle(0);
	initInfo.LegacySingleSrvGpuDescriptor = imguiAllocator->GetGpuHandle(0);
	ImGui_ImplDX12_Init(&initInfo);
}

void PbfApp::PrepareComputeCommandList() {
	// Reset compute allocator and command list for the next compute frame
	DX_API("Failed to reset compute allocator")
		computeAllocator->Reset();
	DX_API("Failed to reset compute list")
		computeList->Reset(computeAllocator.Get(), nullptr);

	// Bind the shared descriptor heap: must be done each time the compute list is reset
	ID3D12DescriptorHeap* computeHeaps[] = { mainAllocator->GetHeap() };
	computeList->SetDescriptorHeaps(1, computeHeaps);
}

void PbfApp::PrepareCommandList() {
	// reset the command allocator, freeing the memory used by the previous frame's commands
	// this can only be done after the GPU finished executing those commands
	DX_API("Failed to reset graphics command allocator")
		commandAllocator->Reset();

	// command list must be reset before we start recording commands into it
	// second param is initial pipeline state, don't need it yet
	DX_API("Failed to reset graphics command list")
		commandList->Reset(commandAllocator.Get(), nullptr);

	// tell the GPU what region of the screen to draw to
	commandList->RSSetViewports(1, &viewPort); // the visible area (full window)
	commandList->RSSetScissorRects(1, &scissorRect); // the clipping rectangle (also full window)

	// get a CPU handle to the depth stencil view (DSV)
	CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsvHeap->GetCPUDescriptorHandleForHeapStart());

	const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };

	if (neuralPass && neuralPass->enabled) {
		// Neural path: scene draws go to the intermediate sceneRT; the swapchain backbuffer
		// is left in PRESENT state and is picked up in phase 2 (RecordPostProcess).
		// sceneRT was created in RENDER_TARGET state and is restored there by RecordPreProcess
		// at the end of every frame, so no barrier is needed here.
		auto rtvHandle = neuralPass->GetSceneRtvHandle();
		commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
		commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	} else {
		// Normal path: transition the current back buffer from PRESENT to RENDER_TARGET
		commandList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(
				renderTargets[swapChainBackBufferIndex].Get(),
				D3D12_RESOURCE_STATE_PRESENT,
				D3D12_RESOURCE_STATE_RENDER_TARGET));

		CD3DX12_CPU_DESCRIPTOR_HANDLE rHandle(
			rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			swapChainBackBufferIndex,
			rtvDescriptorHandleIncrementSize);
		commandList->OMSetRenderTargets(1, &rHandle, FALSE, &dsvHandle);
		commandList->ClearRenderTargetView(rHandle, clearColor, 0, nullptr);
	}

	// clear the depth buffer to 1.0 (maximum depth = far plane)
	commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	// make the SRV heap visible to the GPU for this command list, so shaders can access textures in it
	ID3D12DescriptorHeap* graphicsHeaps[] = { mainAllocator->GetHeap() };
	commandList->SetDescriptorHeaps(1, graphicsHeaps);
}

void PbfApp::ExecuteGraphics() {
	//transition the back buffer back to "present" state so the swap chain can display it
	commandList->ResourceBarrier(1, // number of barriers
		&CD3DX12_RESOURCE_BARRIER::Transition( // helper function to create a transition barrier
			renderTargets[swapChainBackBufferIndex].Get(), // resource: the current back buffer, identified by the swap chain's current back buffer index
			D3D12_RESOURCE_STATE_RENDER_TARGET, // before: we just rendered into the back buffer
			D3D12_RESOURCE_STATE_PRESENT)); // after: we want to present the back buffer

	// close the command list, no more commands can be recorded until the next Reset()
	DX_API("Failed to close command list")
		commandList->Close();
	ID3D12CommandList* graphicsCls[] = { commandList.Get() };
	commandQueue->ExecuteCommandLists(_countof(graphicsCls), graphicsCls);

	DX_API("Failed to present swap chain")
		swapChain->Present(0, 0);
}

void PbfApp::ExecuteCompute() {
	DX_API("Failed to close compute list")
		computeList->Close();
	ID3D12CommandList* computeCls[] = { computeList.Get() };
	computeCommandQueue->ExecuteCommandLists(_countof(computeCls), computeCls);
}

void PbfApp::flipDoubleBuffers() {
	// - particleFieldDBs: back (sorted by permutateCS) becomes the new front for compute
	// - snapshot DBs: back (written by compute) becomes the new front for graphics
	// - lod depth DB: back (written by graphics) becomes the new front for compute
	for (UINT f = 0; f < PF_COUNT; f++) particleFieldDB[f]->flip();
	positionSnapshotDB->flip();
	densitySnapshotDB->flip();
	lodSnapshotDB->flip();
	cellCountSnapshotDB->flip();
	cellPrefixSumSnapshotDB->flip();
	lod->GetParticleDepthDB()->flip();
	sbdPositionSnapshotDB->flip();
}

// Override Render() to decouple physics (compute queue) from graphics (direct queue).
//
// Physics step: CPU waits for compute frame N-1 to finish (so the allocator can be reused),
// records compute frame N onto computeList, submits it, signals computeFence to N.
//
// Graphics step: graphics queue GPU-waits on frame N-1
// to ensure the snapshot is ready, records and submits the scene draw, presents,
// signals graphicsFence, and CPU-waits on it.
void PbfApp::Render() {
	frameCount++; // increment N for this next render
	Throttle(); // apply fps cap if necessary
	t0 = std::chrono::high_resolution_clock::now(); // debug time measurement start

	if (physicsRunning) {
		// Wait for compute frame N-1 before reusing the allocator and readback buffers.
		cpuWaitForCompute(frameCount - 1);

		// Both compute AND graphics frame N-1 are done at this point (graphics waited at end
		// of last Render()). Safe to flip all double buffers:
		flipDoubleBuffers();

		CalculateAvgDensity();
		CalculateAvgLod();

		UpdateComputeCb(lastDt);

		PrepareComputeCommandList();
		RecordComputeCommands();
		ExecuteCompute();
	}
	// signal that after the compute queue reaches this point, the particle data
	// for frame N is ready
	computeFence.signal(computeCommandQueue, frameCount); 

	// GPU-stall graphics until compute frame N-1's snapshot writes are complete.
	graphicsWaitForCompute(frameCount - 1);

	PrepareCommandList();
	RecordGraphicsCommands();

	if (captureNextFrame && neuralPass && neuralPass->enabled) {
		RecordCapture();
		captureNextFrame = false;
	}

	if (neuralPass && neuralPass->enabled) {
		// ---- Phase 1: scene draws + toNchwCS conversion ----
		// RecordPreProcess dispatches toNchwCS (sceneRT → dmlInputBuf) and handles barriers.
		// Leaves dmlInputBuf in COMMON (ready for ORT) and sceneRT in RENDER_TARGET (next frame).
		neuralPass->RecordPreProcess(commandList.Get());

		DX_API("Failed to close phase-1 command list")
			commandList->Close();
		ID3D12CommandList* phase1[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(phase1), phase1);

		// ---- ORT inference: DML work queued on commandQueue, FIFO after phase 1 ----
		neuralPass->RunInference();

		// ---- Phase 2: fromNchwCS → backbuffer, then ImGui overlay ----
		// commandList->Reset() reuses the allocator without resetting it (safe: GPU is still
		// running phase 1, but we don't reset the allocator until the start of the next frame).
		DX_API("Failed to reset command list for phase 2")
			commandList->Reset(commandAllocator.Get(), nullptr);
		commandList->RSSetViewports(1, &viewPort);
		commandList->RSSetScissorRects(1, &scissorRect);

		// RecordPostProcess: fromNchwCS (dmlOutputBuf → outputTex) + CopyResource to backbuffer.
		// Leaves backbuffer in RENDER_TARGET for the ImGui draws below.
		neuralPass->RecordPostProcess(
			commandList.Get(), renderTargets[swapChainBackBufferIndex].Get());

		// Bind backbuffer as the render target for ImGui (no DSV needed for UI).
		CD3DX12_CPU_DESCRIPTOR_HANDLE bbRtv(
			rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			swapChainBackBufferIndex,
			rtvDescriptorHandleIncrementSize);
		commandList->OMSetRenderTargets(1, &bbRtv, FALSE, nullptr);
		BuildImGui();
	} else {
		BuildImGui();
	}

	// ExecuteGraphics transitions backbuffer RENDER_TARGET → PRESENT, closes the command list
	// (phase 2 in the neural path, the only list in the normal path), and presents.
	ExecuteGraphics();

	// Signal graphicsFence and CPU-wait: blocks until the graphics queue (including the
	// GPU-side wait above and all subsequent draws) finishes, meaning that the graphics
	// command queue has processed the commands that render frame N-1. After this the graphics
	// allocator is safe to reset next frame, and render frame N.
	graphicsFence.signal(commandQueue, frameCount - 1);
	cpuWaitForGraphics(frameCount - 1);

	SaveCaptureIfPending();

	// save debug timer value for display in ImGui
	t1 = std::chrono::high_resolution_clock::now();
	debugTimer = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

void PbfApp::RecordComputeCommands() {
	UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	// predictShader is the only shader that runs before the sort
	// it has to, of course, because the sort is based on the *predicted* positions,
	// which only exist after this shader has ran. However, since the sort is what produces
	// the data in the back buffers, predictShader must write its outputs (predictedPosition)
	// to the front buffers, which are then sorted into the back buffers by spatialGrid->Build(). 
	// All subsequent shaders read from and write to the back buffers.
	predictShader->dispatch_then_barrier(computeList.Get(), numGroups);

	spatialGrid->Build(computeList.Get());

	if (sbdRunning) RecordSbdGridBuild(computeList.Get());

	collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	lod->CalculateLod(computeList.Get());

	// Snapshot LOD before the solver loop decrements it. Written into the back LOD snapshot so
	// that flip() in the next Render() promotes it to front for the graphics queue to read.
	{
		GpuBuffer::P lodBuf = lod->GetLodBuffer();
		lodBuf->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
		lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
		computeList->CopyBufferRegion(lodSnapshotDB->getBack()->Get(), 0,
			lodBuf->Get(), 0, (UINT64)numParticles * sizeof(UINT));
		computeList->CopyBufferRegion(lodReadbackBuffer->Get(), 0,
			lodBuf->Get(), 0, (UINT64)numParticles * sizeof(UINT));
		lodBuf->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
		lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	}

	for (int iter = 0; iter < solverIterations; iter++) {
		(gsmEnabled ? gsmLambdaShader : lambdaShader)->dispatch_then_barrier(computeList.Get(), numGroups);
		(gsmEnabled ? gsmDeltaShader : deltaShader)->dispatch_then_barrier(computeList.Get(), numGroups);
		positionFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups);
		collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}

	if (sbdRunning) sbdPoreSuctionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	updateVelocityShader->dispatch_then_barrier(computeList.Get(), numGroups);
	(gsmEnabled ? gsmVorticityShader : vorticityShader)->dispatch_then_barrier(computeList.Get(), numGroups);
	(gsmEnabled ? gsmConfinementViscosityShader : confinementViscosityShader)->dispatch_then_barrier(computeList.Get(), numGroups);
	velocityFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups);
	updatePositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	if (sbdNeedsReset) {
		sbdNeedsReset = false;
		const UINT64 posBytes = (UINT64)numSbdNodes * sizeof(Float3);
		sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
		computeList->CopyBufferRegion(sbdFieldBuffers[SBD_POSITION]->Get(), 0,
			sbdPositionUploadBuffer->Get(), 0, posBytes);
		sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
		sbdFieldBuffers[SBD_VELOCITY]->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
		computeList->CopyBufferRegion(sbdFieldBuffers[SBD_VELOCITY]->Get(), 0,
			sbdVelocityUploadBuffer->Get(), 0, posBytes);
		sbdFieldBuffers[SBD_VELOCITY]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	}

	if (sbdRunning) {
		UINT sbdGroups = ((UINT)numSbdNodes + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
		sbdPredictShader->dispatch_then_barrier(computeList.Get(), sbdGroups);

		// 24 strain passes: 12 orientations × 2 parities.
		// orion bit layout: [1:0]=axes.x [3:2]=axes.y [5:4]=axes.z [6]=signY [7]=signZ [8]=parity
		// Each row of 'orientations' in the shader: |a| gives axis index, sign(a) gives the sign bit.
		// sign convention: bit=1 means positive (>=0), bit=0 means negative.
		static const UINT sbdOrionValues[24] = {
			// parity 0 — 12 orientations

			0b011100100,
			0b000100100,
			0b001011000,
			0b010011000,

			0b011001001,
			0b000001001,
			0b001100001,
			0b010100001,

			0b011010010,
			0b000010010,
			0b001000110,
			0b010000110,

			0b111100100,
			0b100100100,
			0b101011000,
			0b110011000,

			0b111001001,
			0b100001001,
			0b101100001,
			0b110100001,

			0b111010010,
			0b100010010,
			0b101000110,
			0b110000110,
//			228,  // ( 0, +1, +2)  axes=(0,1,2) signs=(+,+)
//			 88,  // ( 0, +2, -1)  axes=(0,2,1) signs=(+,-)
//			 36,  // ( 0, -1, -2)  axes=(0,1,2) signs=(-,-)
//			152,  // ( 0, -2, +1)  axes=(0,2,1) signs=(-,+)
//			210,  // (+2,  0, +1)  axes=(2,0,1) signs=(+,+)
//			225,  // (-1,  0, +2)  axes=(1,0,2) signs=(+,+)
//			 82,  // (-2,  0, -1)  axes=(2,0,1) signs=(+,-)
//			 97,  // (+1,  0, -2)  axes=(1,0,2) signs=(+,-)
//			201,  // (+1, +2,  0)  axes=(1,2,0) signs=(+,+)
//			134,  // (+2, -1,  0)  axes=(2,1,0) signs=(-,+)
//			137,  // (-1, -2,  0)  axes=(1,2,0) signs=(-,+)
//			198,  // (-2, +1,  0)  axes=(2,1,0) signs=(+,+)
//			// parity 1 — same 12 orientations with bit 8 set (+256)
//			484, 344, 292, 408, 466, 481, 338, 353, 457, 390, 393, 454,
		};
		for (UINT iSbdOrionIndex = 0; iSbdOrionIndex < 24; iSbdOrionIndex++)
		{
			UINT orion = sbdOrionValues[iSbdOrionIndex];
			if (!sbdOrionFullOrbit) {
				orion = sbdOrionValues[sbdOrionIndex];
			}
			UINT sbdStrainCellsPerCall = (SBD_DIM_X / 2) * SBD_DIM_Y * SBD_DIM_Z;
			if ((orion & 0x3) == 1)
				sbdStrainCellsPerCall = SBD_DIM_X * (SBD_DIM_Y / 2) * SBD_DIM_Z;
			if ((orion & 0x3) == 2)
				sbdStrainCellsPerCall = SBD_DIM_X * SBD_DIM_Y * (SBD_DIM_Z / 2);
			UINT sbdStrainGroups = ((UINT)sbdStrainCellsPerCall + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
			sbdStrainShader->dispatch_then_barrier_with_constant(computeList.Get(), sbdStrainGroups, 2, orion);

			if (!sbdOrionFullOrbit) break;
		}

		sbdUpdateVelocityShader->dispatch_then_barrier(computeList.Get(), sbdGroups);
	}

	WriteSnapshot();
}

void PbfApp::WriteSnapshot() {
	// Copy position and density into the back snapshot buffers.
	// After this frame's flip(), back becomes the new front for graphics to read.
	// Sorted particle data lives in back (permutateCS wrote there; flip happens CPU-side in Render()).
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	particleFieldDB[PF_DENSITY]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());

	computeList->CopyBufferRegion(positionSnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_POSITION]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(Float3));
	computeList->CopyBufferRegion(densitySnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_DENSITY]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(float));

	computeList->CopyBufferRegion(densityReadbackBuffer->Get(), 0,
		particleFieldDB[PF_DENSITY]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(float));

	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	particleFieldDB[PF_DENSITY]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());

	// Copy grid buffers into the back grid snapshot slots.
	spatialGrid->GetCellCountBuffer()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	spatialGrid->GetPrefixSumBuffer()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());

	const UINT64 gridBufSize = (UINT64)numCells * sizeof(UINT);
	computeList->CopyBufferRegion(cellCountSnapshotDB->getBack()->Get(), 0,
		spatialGrid->GetCellCountBuffer()->Get(), 0, gridBufSize);
	computeList->CopyBufferRegion(cellPrefixSumSnapshotDB->getBack()->Get(), 0,
		spatialGrid->GetPrefixSumBuffer()->Get(), 0, gridBufSize);

	spatialGrid->GetCellCountBuffer()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	spatialGrid->GetPrefixSumBuffer()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());

	// Copy SBD positions to the back snapshot so flip() makes them available to graphics.
	if (sbdRunning) {
		sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
		sbdPositionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
		computeList->CopyBufferRegion(sbdPositionSnapshotDB->getBack()->Get(), 0,
			sbdFieldBuffers[SBD_POSITION]->Get(), 0, (UINT64)numSbdNodes * sizeof(Float3));
		sbdFieldBuffers[SBD_POSITION]->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
		sbdPositionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	}
}

void PbfApp::RecordGraphicsCommands() {
	backgroundMesh->Draw(commandList.Get());
	for (int i = 0; i < NUM_OBSTACLES; i++)
		obstacles[i]->Draw(commandList.Get());

	// Draw SBD nodes as point-sprite spheres alongside the fluid.
	// sbdPositionSnapshotDB front is in NON_PIXEL_SHADER_RESOURCE (home state), sufficient for VS access.
	sbdMesh->Draw(commandList.Get());

	// Promote positionSnapshot front to pixel-visible before any draw that uses it from the pixel stage.
	constexpr D3D12_RESOURCE_STATES SRV_ALL =
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	if (shadingMode == ShadingMode::LIQUID)
		positionSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());

	if (lod->mode == LodSystem::Mode::DTVS) { // DTVS requires depth data
		// DrawParticleDepth restores the RTV after its depth-only draw via the last argument.
		// When the neural pass is active the destination is sceneRT, not the swapchain backbuffer.
		D3D12_CPU_DESCRIPTOR_HANDLE mainRtv = (neuralPass && neuralPass->enabled)
			? neuralPass->GetSceneRtvHandle()
			: CD3DX12_CPU_DESCRIPTOR_HANDLE(
				rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
				swapChainBackBufferIndex,
				rtvDescriptorHandleIncrementSize);
		lod->DrawParticleDepth(
			commandList.Get(),
			perFrameCb.GetGPUVirtualAddress(),
			mainAllocator->GetGpuHandle(particleSrvTableStart),
			dsvHeap->GetCPUDescriptorHandleForHeapStart(),
			mainRtv);
	}		

	if (shadingMode == ShadingMode::LIQUID) {
		DrawLiquidSurface();
		// Transition position snapshot back to NON_PIXEL_SHADER_RESOURCE so that the next non-liquid 
		// draw can read it from compute shaders if needed.
		positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	}
	else {
		particleMesh->Draw(commandList.Get());
	}
}

// Fill the density volume and draw the ray-marched liquid surface, all on the graphics command list.
// densityVolumeCS is dispatched here (graphics queue) reading from the previous frame's position and
// grid snapshots at [readIdx]. liquidPS then ray-marches through the freshly filled volume in the same frame.
// The GPU-wait at the front of the graphics list (graphicsWaitForCompute(N-1)) guarantees that
// snapshotPosition[readIdx] and the grid snapshots are fully written by compute before this runs.
void PbfApp::DrawLiquidSurface() {
	densityVolume->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());

	// Splat: one thread per particle; each writes Poly6 to densityVolume via CAS float atomic add.
	// dispatch_then_barrier emits a UAV barrier on densityVolume (it's in outputs), ensuring all
	// splat writes are visible to the liquidPS SRV read that follows.
	UINT numGroups = ((UINT)numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
	splatDensityShader->dispatch_then_barrier(commandList.Get(), numGroups);

	// Transition densityVolume UAV -> combined SRV state for liquidPS.
	// Both NON_PIXEL_SHADER_RESOURCE and PIXEL_SHADER_RESOURCE bits must be set:
	// D3D12 validates that descriptor-table SRVs carry both flags (#538 error otherwise).
	constexpr D3D12_RESOURCE_STATES SRV_ALL =
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	densityVolume->Transition(SRV_ALL, commandList.Get());
	cellCountSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());

	liquidMesh->Draw(commandList.Get());

	cellCountSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	densityVolume->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());

	// Clear the density volume for the next frame.
	const UINT clearVal[4] = { 0u, 0u, 0u, 0u };
	commandList->ClearUnorderedAccessViewUint(
		densityVolumeHandle, densityVolClearCpuHandle,
		densityVolume->Get(), clearVal, 0, nullptr);

	densityVolume->Transition(D3D12_RESOURCE_STATE_COMMON, commandList.Get());
}

void PbfApp::RecordCapture() {
	auto* rt = neuralPass->GetSceneRT();
	auto  d  = rt->GetDesc();
	UINT  width = (UINT)d.Width;
	UINT  height = (UINT)d.Height;
	UINT  rowPitch = (width * 4u + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u) &
	                 ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);

	if (!captureReadback || captureRowPitch != rowPitch) {
		captureRowPitch = rowPitch;
		CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_READBACK);
		auto bd = CD3DX12_RESOURCE_DESC::Buffer((UINT64)rowPitch * height);
		DX_API("Failed to create capture readback buffer")
			device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
				IID_PPV_ARGS(captureReadback.ReleaseAndGetAddressOf()));
		captureReadback->SetName(L"CaptureReadback");
	}

	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE));

	D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
	fp.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, rowPitch };
	CD3DX12_TEXTURE_COPY_LOCATION src(rt, 0);
	CD3DX12_TEXTURE_COPY_LOCATION dst(captureReadback.Get(), fp);
	commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
		rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET));

	pendingReadback = true;
}

void PbfApp::SaveCaptureIfPending() {
	if (!pendingReadback) return;
	pendingReadback = false;

	auto* rt = neuralPass->GetSceneRT();
	auto  d  = rt->GetDesc();
	UINT  width  = (UINT)d.Width, height = d.Height;

	std::filesystem::create_directories(captureInputDir);

	char fname[16]; sprintf_s(fname, "%04d.png", captureIndex++);
	std::string  path  = captureInputDir + "\\" + fname;
	std::wstring wpath(path.begin(), path.end());

	D3D12_RANGE readRange = { 0, (SIZE_T)captureRowPitch * height };
	uint8_t*    pixels    = nullptr;
	captureReadback->Map(0, &readRange, reinterpret_cast<void**>(&pixels));

	// Write RGBA PNG via WIC (windowscodecs.dll is included with Windows; no extra DLLs needed).
	HRESULT cohr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	{
		IWICImagingFactory*    wic   = nullptr;
		IWICStream*            stm   = nullptr;
		IWICBitmapEncoder*     enc   = nullptr;
		IWICBitmapFrameEncode* frame = nullptr;
		CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
		wic->CreateStream(&stm);
		stm->InitializeFromFilename(wpath.c_str(), GENERIC_WRITE);
		wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc);
		enc->Initialize(stm, WICBitmapEncoderNoCache);
		enc->CreateNewFrame(&frame, nullptr);
		frame->Initialize(nullptr);
		frame->SetSize(width, height);
		WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppRGBA;
		frame->SetPixelFormat(&fmt);
		frame->WritePixels(height, captureRowPitch, captureRowPitch * height, pixels);
		frame->Commit();
		enc->Commit();
		frame->Release(); enc->Release(); stm->Release(); wic->Release();
	}
	if (cohr == S_OK) CoUninitialize();

	D3D12_RANGE writeRange = { 0, 0 };
	captureReadback->Unmap(0, &writeRange);

	OutputDebugStringA(("Capture saved → " + path + "\n").c_str());
}

void PbfApp::BuildImGui() {
	// begin a new ImGui frame, which gives us a clean slate to construct the UI for this frame
	ImGui_ImplDX12_NewFrame(); // tell ImGui about the new frame for DX12
	ImGui_ImplWin32_NewFrame(); // tell ImGui about the new frame for Win32 (input handling, time, etc)
	// the core library consumes the input state the backends just wrote and begins a new frame
	ImGui::NewFrame(); // after this we can create ImGui widgets for this frame

	// InputFloat/InputInt: text field with +/- stepper buttons. Type a value and press Enter.
	// The "step" argument is how much the +/- buttons change the value per click.
	// This is the immediate mode paradigm: we construct the same UI every frame, and ImGui handles the state internally.
	// InputFloat/Int reads the current value from the pointer, renders the widget into the draw list, and
	// writes the value back to the pointer if the user changed it
	ImGui::Begin("PBF Controls");

	if (ImGui::CollapsingHeader("Particles", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushItemWidth(100);
		static const char* shadingModeItems[] = { "Unicolor", "Density", "LOD", "Liquid" };
		ImGui::Combo("Shading", &shadingMode, shadingModeItems, IM_ARRAYSIZE(shadingModeItems));
		if (shadingMode == ShadingMode::LIQUID)
			ImGui::InputFloat("Liquid iso threshold", &liquidIsoThreshold, 1.0f, 10.0f, "%.1f");
		static const char* lodModeItems[] = { "Non-adaptive", "DTC", "DTVS" };
		int lodModeInt = (int)lod->mode;
		if (ImGui::Combo("LOD mode", &lodModeInt, lodModeItems, IM_ARRAYSIZE(lodModeItems)))
			lod->mode = (LodSystem::Mode)lodModeInt;
		ImGui::InputInt("Solver iterations", &solverIterations, 1);
		ImGui::InputInt("Min LOD", &minLOD, 1);
		ImGui::InputFloat("Epsilon (relaxation)", &epsilon, 0.5f, 1.0f, "%.2f");
		ImGui::InputFloat("Viscosity (XSPH)", &viscosity, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Artificial pressure", &sCorrK, 0.005f, 0.05f, "%.4f");
		ImGui::InputFloat("Vorticity epsilon", &vorticityEpsilon, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Adhesion", &adhesion, 0.01f, 0.1f, "%.3f");
		ImGui::PopItemWidth();
		ImGui::Checkbox("Fountain", &fountainEnabled);
		ImGui::SameLine();
		ImGui::Checkbox("FPS cap", &fpsCapped);
		ImGui::SameLine();
		ImGui::Checkbox("GSM", &gsmEnabled);
		ImGui::Spacing();
		ImGui::Text("%d particles, %u cells", numParticles, gridDim * gridDim * gridDim);
		ImGui::Text("%.1f FPS, render: %.2f ms", ImGui::GetIO().Framerate, debugTimer);
		ImGui::Text("avg density: %.2f (rho0: %.2f)", avgDensity, rho0);
		ImGui::Text("avg LOD: %.2f", avgLod);
	}

	if (ImGui::CollapsingHeader("Objects")) {
		const char* obstacleNames[NUM_OBSTACLES];
		for (int i = 0; i < NUM_OBSTACLES; i++)
			obstacleNames[i] = obstacles[i]->GetName();
		ImGui::Combo("Obstacle", &selectedObstacle, obstacleNames, NUM_OBSTACLES);
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Pos", &obstacles[selectedObstacle]->position.x, 0.1f);
		ImGui::DragFloat3("Rot (deg)", &obstacles[selectedObstacle]->eulerDeg.x, 1.0f);
		ImGui::DragFloat("Scale", &obstacles[selectedObstacle]->scale, 0.01f, 0.01f, 100.0f);
		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Lights")) {
		char nameBufs[NUM_LIGHTS][12];
		const char* namePtrs[NUM_LIGHTS];
		for (int i = 0; i < NUM_LIGHTS; i++) {
			snprintf(nameBufs[i], sizeof(nameBufs[i]), "Light %d", i);
			namePtrs[i] = nameBufs[i];
		}
		ImGui::Combo("Light", &selectedLight, namePtrs, NUM_LIGHTS);
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Direction", &lightDirs[selectedLight].x, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color", &lightColors[selectedLight].x);
		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Bounding Box")) {
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Box min", &boxMin.x, 0.1f, gridMin.x, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat3("Box max", &boxMax.x, 0.1f, 0.0f, gridMax.x, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Soft Body")) {
		ImGui::Checkbox("SBD running", &sbdRunning);
		ImGui::SameLine();
		if (ImGui::Button("Reset")) sbdNeedsReset = true;
		if (ImGui::Checkbox("Full Orbit", &sbdOrionFullOrbit)) {
			sbdOrionFullOrbit = true;
		}
		ImGui::SliderInt("Orientation", &sbdOrionIndex, 0, 23);
		ImGui::SliderFloat("Suction Strength", &sbdSuctionStrength, 0.0f, 2.0f, "%.4f");
		ImGui::TextDisabled("%d BCC nodes (%dx%dx%d * 2)", numSbdNodes, SBD_DIM_X, SBD_DIM_Y, SBD_DIM_Z);
		ImGui::TextDisabled("Shaders: sbdPredictCS, sbdStrainCS, sbdUpdateVelocityCS");
	}

	if (ImGui::CollapsingHeader("Neural Post-Process")) {
		if (neuralPass && neuralPass->enabled) {
			ImGui::Checkbox("U-Net enabled", &neuralPass->enabled);
			ImGui::TextDisabled("TinyUNet ~93K params, DirectML on GPU");
			ImGui::Separator();
			ImGui::TextUnformatted("Training capture");
			if (ImGui::Button("Capture input frame")) captureNextFrame = true;
			ImGui::SameLine();
			ImGui::Text("%d saved", captureIndex);
			ImGui::TextDisabled("%s", captureInputDir.c_str());
			ImGui::TextDisabled("Train: python models/train_unet.py --data captures");
		} else if (neuralPass) {
			UINT rW = (UINT)scissorRect.right, rH = (UINT)scissorRect.bottom;
			bool mismatch = neuralPass->modelWidth > 0 &&
				(neuralPass->modelWidth != rW || neuralPass->modelHeight != rH);
			if (mismatch) {
				ImGui::TextColored(ImVec4(1, 0.4f, 0, 1), "Resolution mismatch!");
				ImGui::TextDisabled("Model: %ux%u  Render: %ux%u",
					neuralPass->modelWidth, neuralPass->modelHeight, rW, rH);
				ImGui::TextDisabled("Re-export: python models/export_unet.py --width %u --height %u", rW, rH);
			} else {
				ImGui::Checkbox("U-Net enabled", &neuralPass->enabled);
			}
		} else {
			ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No model loaded");
			ImGui::TextDisabled("Run: python models/export_unet.py");
			ImGui::TextDisabled("Place unet_postprocess.onnx in Bin\\models\\");
		}
	}

	ImGui::End();

	// Finalizes the frame.ImGui takes all the widgets you defined since NewFrame(), performs layout
	// (positions, sizes, clipping), and produces an ImDrawData structure : a list of vertex buffers, index
	// buffers, and draw commands that describe exactly what triangles to draw and with what textures.No
	// GPU calls happen here - it's pure CPU-side geometry generation.
	ImGui::Render();
	// ImGui needs its own SRV heap bound (for the font texture), so we switch heaps here.
	// The scene's srvHeap was used during RecordGraphicsCommands; that's done, so this is safe.
	ID3D12DescriptorHeap* imguiHeaps[] = { imguiAllocator->GetHeap() };
	commandList->SetDescriptorHeaps(1, imguiHeaps);
	// This is where ImGui's geometry actually gets drawn. GetDrawData() returns the ImDrawData that
	// Render() produced.The D3D12 backend takes it and :
	//  1. Selects this frame's rotating vertex/index buffer pair (alternating between 2 sets for double buffering)
	//	2. Maps the upload buffers and copies ImGui's vertex + index data into them
	//	3. Sets its own root signature and PSO on the command list
	//	4. Sets the viewport, blend factor, and stencil ref
	//	5. For each draw command : sets the scissor rect(ImGui uses scissor for clipping), binds the font
	//		texture SRV, and issues an indexed draw call
	//
	//	After this returns, the command list contains all the triangles needed to render the UI panel, text,
	//	and input fields on top of our scene.
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList.Get());
}

void PbfApp::UpdateExternalForce() {
	// build a horizontal acceleration vector from held arrow keys
	// left/right push along X, up/down push along Z
	externalForce = Float3(0.0f, 0.0f, 0.0f);
	if (arrowLeft) externalForce.x -= externalAcceleration;
	if (arrowRight) externalForce.x += externalAcceleration;
	if (arrowUp) externalForce.z += externalAcceleration;
	if (arrowDown) externalForce.z -= externalAcceleration;
}

void PbfApp::UpdatePerFrameCb() {
	perFrameCb->viewProjTransform = // calculate the combined view-projection matrix and store it in the constant buffer
		camera->GetViewMatrix() * // view matrix: world space -> camera space
		camera->GetProjMatrix(); // projection matrix: camera space -> clip space
	perFrameCb->rayDirTransform = camera->GetRayDirMatrix(); // clip-space coords -> world-space view direction
	perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
	for (int i = 0; i < NUM_LIGHTS; i++) {
		perFrameCb->lights[i].direction = Float4(lightDirs[i], 0.0f);
		perFrameCb->lights[i].color     = Float4(lightColors[i], 0.0f);
	}
	perFrameCb->particleParams = Float4(rho0, 0.0f, 0.0f, PARTICLE_RADIUS); // x = rho0 (for density coloring in PS), w = particle display radius (for billboard sizing in GS)
	perFrameCb->shadingMode = (UINT)shadingMode;
	perFrameCb->minLOD = (UINT)minLOD;
	perFrameCb->maxLOD = (UINT)solverIterations;
	// bbMin.xyz = adjustable collision box min; bbMin.w = liquid density iso-surface threshold.
	// bbMax.xyz = collision box max; bbMax.w unused.
	perFrameCb->bbMin = Float4(boxMin, liquidIsoThreshold);
	perFrameCb->bbMax = Float4(boxMax, 0.0f);
	perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
}

void PbfApp::UpdateComputeCb(float dt) {
	computeCb->dt = dt;
	computeCb->numParticles = numParticles;
	computeCb->sCorrK = sCorrK;
	computeCb->vorticityEpsilon = vorticityEpsilon;
	computeCb->boxMin = boxMin;
	computeCb->epsilon = epsilon;
	computeCb->boxMax = boxMax;
	computeCb->viscosity = viscosity;
	computeCb->externalForce = externalForce;
	computeCb->fountainEnabled = fountainEnabled ? 1 : 0;
	computeCb->adhesion = adhesion;
	for (int i = 0; i < NUM_OBSTACLES; i++) {
		computeCb->obstacles[i].invTransform = obstacles[i]->GetInvTransform();
		computeCb->obstacles[i].sdfMin = obstacles[i]->GetSdfMin();
		computeCb->obstacles[i].sdfMax = obstacles[i]->GetSdfMax();
	}
	computeCb->cameraPos = camera->GetEyePosition();
	computeCb->minLOD = (UINT)minLOD;
	computeCb->maxLOD = (UINT)solverIterations;
	computeCb->viewProjTransform = camera->GetViewMatrix() * camera->GetProjMatrix();
	computeCb->viewportWidth = (float)scissorRect.right;
	computeCb->viewportHeight = (float)scissorRect.bottom;
	computeCb->pushRadius = (shadingMode == ShadingMode::LIQUID) ? 0.0f : PUSH_RADIUS;
	computeCb->sbdSuctionStrength = sbdSuctionStrength;
	computeCb.Upload();
}

void PbfApp::Update(float dt, float T)  {
	camera->Animate(dt); // real dt for responsive camera
	lastDt = std::min(dt, 1.0f / 25.0f); // cap at 25Hz: prevents energy spikes on window drag or stutter
	UpdateExternalForce();
	UpdatePerFrameCb();
	SetObstacleTransforms();
}

void PbfApp::CalculateAvgDensity() {
	// map the readback buffer to CPU memory and copy the density data into a vector
	const UINT64 bufferSize = numParticles * sizeof(float);
	void* pData; // ptr will be set by Map to point at the readback buffer's CPU visible memory
	CD3DX12_RANGE readRange(0, bufferSize);
	// in the Map call, we map with the range we intend to read
	if (SUCCEEDED(densityReadbackBuffer->Get()->Map(0, &readRange, &pData))) { // prepare pData for reading
		memcpy(densityReadbackData.data(), pData, bufferSize); // actual data movement call
		// during the unmap, we unmap while indicating which bytes we dirtied
		CD3DX12_RANGE writeRange(0, 0); // in this case, we wrote nothing
		densityReadbackBuffer->Get()->Unmap(0, &writeRange); // release mapping: invalidate pData
	}

	// Compute average density from readback data, only every AVG_COARSENESS particles to save CPU time.
	double densitySum = 0.0;
	int cnt = 0;
	for (int i = 0; i < numParticles; i += AVG_COARSENESS) {
		densitySum += densityReadbackData[i];
		cnt++;
	}

	avgDensity = static_cast<float>(densitySum / cnt);
}

void PbfApp::CalculateAvgLod() {
	// map the readback buffer to CPU memory and copy the LOD data into a vector
	const UINT64 bufferSize = numParticles * sizeof(uint32_t);
	void* pData; // ptr will be set by Map to point at the readback buffer's CPU visible memory
	CD3DX12_RANGE readRange(0, bufferSize);
	// in the Map call, we map with the range we intend to read
	if (SUCCEEDED(lodReadbackBuffer->Get()->Map(0, &readRange, &pData))) { // prepare pData for reading
		memcpy(lodReadbackData.data(), pData, bufferSize); // actual data movement call
		CD3DX12_RANGE writeRange(0, 0); // in this case, we wrote nothing
		lodReadbackBuffer->Get()->Unmap(0, &writeRange); // release mapping: invalidate pData
	}

	// Compute average LOD from readback data, only every AVG_COARSENESS particles to save CPU time.
	double lodSum = 0.0;
	int cnt = 0;
	for (int i = 0; i < numParticles; i += AVG_COARSENESS) {
		lodSum += lodReadbackData[i];
		cnt++;
	}

	avgLod = static_cast<float>(lodSum / cnt);
}

// This function cannot be called more than once every targetPeriod time: rate limit
// a better way of doing this would be a fixed timestep accumulation, where we decouple
// physics dt from render dt entirely, accumulate wall-clock time, and step physics at a 
// fixed interval
void PbfApp::Throttle() {
	if (fpsCapped) {
		auto deadline = lastFrame + targetPeriod;
		auto remaining = deadline - clock::now();

		// Sleep for all but the last ~1ms to avoid overshooting
		if (remaining > std::chrono::milliseconds(1))
			std::this_thread::sleep_for(remaining - std::chrono::milliseconds(1));

		// Spin-wait the remainder for precision
		while (clock::now() < deadline) {}
	}
	lastFrame = clock::now();
}

void PbfApp::RunNTimes(int n, bool physicsEnabled) {
	bool wasPhysicsRunning = physicsRunning; // save current state to restore later
	physicsRunning = physicsEnabled;
	for (int i = 0; i < n; ++i) {
		Run();
	}
	physicsRunning = wasPhysicsRunning; // restore original state
}

void PbfApp::ReleaseSwapChainResources()  {
	if (neuralPass) neuralPass->ReleaseResolutionResources();
	AsyncComputeApp::ReleaseSwapChainResources();
}

void PbfApp::Resize(int width, int height) {
	cpuWaitForCompute(frameCount);
	cpuWaitForGraphics(frameCount - 1);
	Egg::App::Resize(width, height);
}

void PbfApp::ShutdownImGui() {
	// Teardown in reverse order of initialization :
	// 1. ImGui_ImplDX12_Shutdown() - releases all D3D12 objects the backend created(PSOs, root
	//	  signatures, vertex / index buffers, command allocator, command list, font texture + its SRV)
	// 2. ImGui_ImplWin32_Shutdown() - unhooks from the window, clears input state
	// 3. ImGui::DestroyContext() - frees the global context(GImGui), setting it to nullptr.This is why
	//	  the GetCurrentContext() != nullptr guard in WindowProcess is necessary - messages arriving after
	//    this point must not call into ImGui.
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

// Forward window messages (keyboard, mouse) to the camera, and handle app-level hotkeys
void PbfApp::ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	camera->ProcessMessage(hWnd, uMsg, wParam, lParam);

	if (uMsg == WM_KEYDOWN && wParam == VK_SPACE)
		physicsRunning = !physicsRunning; // toggle physics simulation on/off

	// track arrow key held state for continuous box translation in Update()
	if (uMsg == WM_KEYDOWN) {
		if (wParam == VK_LEFT) arrowLeft = true;
		if (wParam == VK_RIGHT) arrowRight = true;
		if (wParam == VK_UP) arrowUp = true;
		if (wParam == VK_DOWN) arrowDown = true;
	}
	if (uMsg == WM_KEYUP) {
		if (wParam == VK_LEFT) arrowLeft = false;
		if (wParam == VK_RIGHT) arrowRight = false;
		if (wParam == VK_UP) arrowUp = false;
		if (wParam == VK_DOWN) arrowDown = false;
	}
}
