#ifndef SHARED_CONFIG_HLSLI
#define SHARED_CONFIG_HLSLI

// Stringification helpers — used by shaders to embed #define values into root-signature strings.
// Usage: "SRV(t0, numDescriptors = " STR(NUM_OBSTACLES) ")"
// The two-level expansion is required: STR(x) -> XSTR(x) -> #x forces full macro expansion before stringification.
#define XSTR(x) #x
#define STR(x) XSTR(x)

#define THREAD_GROUP_SIZE 256
#define CELL_PER_H 1

// Compile-time simulation constants shared between HLSL and C++.
// These never change at runtime, so keeping them as preprocessor defines
// lets the shader compiler constant-fold all expressions that depend on them,
// and removes them from the constant buffer entirely.

#define PARTICLE_SPACING  0.25f
#define H_MULTIPLIER      2.5f
#define H                 (PARTICLE_SPACING * H_MULTIPLIER)   // SPH smoothing radius

// Rest density: if particles are spaced PARTICLE_SPACING apart, each occupies
// PARTICLE_SPACING^3 volume, so with m=1 the density is 1/d^3.
#define RHO0              (1.0f / (PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING))

// Particle radius used for display and SDF push-out distance.
#define PARTICLE_RADIUS  (PARTICLE_SPACING * 0.4f)
#define PUSH_RADIUS      (PARTICLE_SPACING * 0.4f) // default push distance; C++ uses this to init computeCb.pushRadius

// Artificial pressure reference distance and exponent (Macklin & Muller eq. 13).
#define SCORR_DELTA_Q     (0.2f * H)
#define SCORR_N           4.0f

// Grid dimensions: gridDim cells per axis, each of width H/CELL_PER_H.
#define GRID_DIM          128 // cells per axis; SpatialGrid supports up to 128 (5-pass Blelloch)

// SPH kernel normalization coefficients, precomputed from H.
#define PBF_PI            3.14159265358979323846f
#define POLY6_COEFF       (315.0f / (64.0f * PBF_PI * H*H*H*H*H*H*H*H*H))
#define SPIKY_GRAD_COEFF  (45.0f  / (PBF_PI * H*H*H*H*H*H))

// Grid cell size in each dimension
#define CELL_SIZE H / CELL_PER_H

// Grid world-space half-extent: GRID_DIM cells of width H/CELL_PER_H on each side.
#define BOX_HALF_EXTENT GRID_DIM * CELL_SIZE / 2.0f

// Shading mode constants - must match the order to the ImGui shadingModeItems[]
#define SHADING_UNICOLOR 0
#define SHADING_DENSITY  1
#define SHADING_LOD      2
#define SHADING_LIQUID   3

// Density volume resolution
#define VOXEL_SIZE PARTICLE_SPACING
#define VOL_DIM (uint)(GRID_DIM * CELL_SIZE / VOXEL_SIZE)
#define H_IN_VOXEL H / VOXEL_SIZE

// How coarsely to take the average density and LOD for ImGui display
#define AVG_COARSENESS 100

// Number of solid obstacles in the scene.
// To add a new obstacle: bump this, add an entry to the ObstacleDesc table in
// InitObstacle(), and update numDescriptors in the PredictRootSig /
// CollisionPositionRootSig strings in predictCS.hlsl / collisionPredictedPositionCS.hlsl.
#define NUM_OBSTACLES 1

// Number of directional light sources.
// To add a new light: bump this and add an entry to the lightDirs/lightColors
// initializers in PbfApp.h; the new light will appear in the ImGui "Light" combo.
#define NUM_LIGHTS 3

// BCC soft body dynamics grid dimensions.
// Nodes: sublattice A at integer (i,j,k)*spacing, sublattice B at (i+0.5,j+0.5,k+0.5)*spacing.
// Total node count = 2 * SBD_DIM_X * SBD_DIM_Y * SBD_DIM_Z.
#define SBD_DIM_X 7
#define SBD_DIM_Y 7
#define SBD_DIM_Z 7
#define SBD_NUM_CUBIC_NODES (SBD_DIM_X * SBD_DIM_Y * SBD_DIM_Z)
#define SBD_NUM_NODES (SBD_NUM_CUBIC_NODES + (SBD_DIM_X+1) * (SBD_DIM_Y+1) * (SBD_DIM_Z+1))

// SBD spatial-grid influence radius. Larger than the PBF smoothing radius H so that
// fluid particles can feel suction from SBD nodes even when the BCC lattice spacing (1 m)
// exceeds H. Cell size = SBD_H (one cell per radius), reusing GRID_DIM^3 cells.
#define SBD_H 2.5f

// Number of volume particles seeded inside the animated character mesh.
// Used as bounds check in character grid / density / influence shaders.
#define N_CHAR_PARTICLES 512

#endif