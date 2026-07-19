# Position Based Fluids

A real-time, GPU-driven fluid simulator implementing **Position Based Fluids** (Macklin & Müller, NVIDIA 2013) in **DirectX 12**. The entire simulation — neighbour search, constraint solving, vorticity confinement, viscosity, collision, and level-of-detail — runs on the GPU via compute shaders on a dedicated **async compute queue**, decoupled from rendering. Hundreds of thousands of particles are simulated interactively and rendered as a ray-marched liquid surface.

This project is the subject of my BSc thesis (_CRQEYD Position Based Fluids.pdf_, included in this repository).

Built on top of **Egg**, a small D3D12 teaching framework, using Dear ImGui for the parameter-tuning UI.

---

## Features

- **Position Based Fluids solver** — density-constraint projection with a Newton/Jacobi solver loop, artificial pressure to reduce clumping, XSPH viscosity, and vorticity confinement, following Macklin & Müller.
- **Fully GPU-resident** — every simulation step is a compute dispatch; particle data never round-trips to the CPU (only small averages are read back for the UI).
- **Async compute** — physics runs on a separate compute queue and is double-buffered against the graphics queue, so the simulation is decoupled from vsync.
- **GPU spatial hash grid** — a cubic, power-of-two grid (up to 128³ cells) with counting sort built from a multi-pass Blelloch prefix sum; Morton-code (Z-order) indexing for cache-friendly neighbour lookups.
- **Adaptive Level-of-Detail (LOD)** — per-particle solver iteration counts assigned each frame, with two strategies: Distance-To-Camera / Distance-To-Visible-Surface.
- **Ray-marched liquid surface** — particles are splatted into a 3D density volume and rendered as an iso-surface via ray marching, with reflection/refraction against an environment cubemap.
- **Solid obstacles via signed distance fields** — arbitrary meshes are baked into SDF volumes offline (Python tool) and used for collision push-out; obstacles can be transformed at runtime.
- **Groupshared-memory (GSM) optimized shader variants** — cooperative neighbour loading into groupshared memory for the heaviest passes, toggleable at runtime to compare against the naive baseline.
- **Multiple shading modes** — unicolor, density heatmap, LOD visualization, and the ray-marched liquid surface.
- **Live tuning** — an ImGui panel exposes every physical and rendering parameter, plus a fountain emitter, multiple directional lights, and per-frame diagnostics.

---

## Controls

| Input                   | Action                                       |
| ----------------------- | -------------------------------------------- |
| `W` `A` `S` `D` + mouse | First-person camera                          |
| Arrow keys              | Apply horizontal external force to the fluid |
| `Spacebar`              | Pause / resume the physics simulation        |

**ImGui panel (_PBF Controls_)** — solver iterations, min LOD, relaxation ε, XSPH viscosity, artificial pressure, vorticity ε, adhesion, shading mode, LOD mode, liquid iso-threshold, fountain toggle, FPS cap, GSM toggle, obstacle transforms, and directional lights. Live stats show particle/cell counts, FPS, average density, and average LOD.

---

## Building

**Requirements**

- Windows 10/11 with a Direct3D 12–capable GPU (feature level 12.0)
- Visual Studio 2022 (v143 toolset) with the _Desktop development with C++_ workload and a recent Windows 10/11 SDK

**Steps**

1. Open `Position Based Fluids.sln` in Visual Studio 2022.
2. Select the **Release | x64** configuration.
3. Build and run the **pbf** project (it is the startup project and depends on the bundled `Egg`, `DirectXTex`, `LuaBind`, and `UtilCopyDLL` projects).

Third-party dependencies (Egg, Dear ImGui, DirectXTex, assimp, Boost, Lua/LuaBind, PhysX) are vendored under `Common/`, `Egg/`, and `physx/`, so no package manager setup is required.
