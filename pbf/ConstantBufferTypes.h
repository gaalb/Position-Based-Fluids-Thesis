#pragma once

#include "Egg/Math/Float4x4.h"
#include "SharedConfig.hlsli"

using namespace Egg::Math;

// note on __declspec(align(16)): HLSL constant buffers are read by the GPU in 16-byte chunks ("rows"),
// every field in a cbuffer must fit neatly withing these rows, i.e. an element must not straddle
// a 16-byte boundary. This means two things:
// 1: the struct must be placed in a memory address that's a multiple of 16: this is what declspec does
// 2: the struct' fields must not pass a 16-byte boundary: assured by paddings


// Shading mode indices, used by particlePS and set via ImGui.
// Add new modes here and add a corresponding branch in particlePS.hlsl.
namespace ShadingMode {
	constexpr UINT UNICOLOR = 0; // flat blue base color
	constexpr UINT DENSITY  = 1; // blue->green->red by density relative to rho0 (default)
	constexpr UINT LOD      = 2; // blue->orange by LOD value (minLOD..maxLOD)
	constexpr UINT LIQUID   = 3; // ray-marched liquid surface with Blinn-Phong shading
}

// One directional light source. Layout must match LightData in each HLSL shader.
__declspec(align(16)) struct LightData {
	Float4 direction; // xyz = direction toward light (normalized), w = unused
	Float4 color;     // xyz = RGB * intensity, w = unused
	// 32 bytes per light
};

// per-frame data sent to shaders every frame - camera matrices etc.
// must be 16-byte aligned because the GPU reads constant buffers in 16byte chunks
__declspec(align(16)) struct PerFrameCb {
	Float4x4 viewProjTransform; // offset 0: world space -> clip space  (view * projection)
	Float4x4 rayDirTransform;   // offset 64: NDC -> world space direction  (inverse of view-rotation * projection)
	Float4 cameraPos; // offset 128: camera position in world space, w=1
	LightData lights[NUM_LIGHTS]; // offset 144: array of directional lights (32 bytes each)
	Float4 particleParams; // offset 144+32*NUM_LIGHTS: x = rho0, w = particle display radius
	UINT shadingMode; // which shading branch to use (ShadingMode::*)
	UINT minLOD; // minimum LOD value (for LOD color normalization)
	UINT maxLOD; // maximum LOD value (for LOD color normalization)
	float _pad; // padding to next 16-byte boundary
	Float4 bbMin; // xyz = adjustable boxMin, w = liquid density iso-surface threshold
	Float4 bbMax; // xyz = adjustable boxMax, w = unused
};

// per-draw data for the solid obstacle rendering shader
__declspec(align(16)) struct SolidCb {
	Float4x4 modelMat; // object-to-world transform for solidVS
};

// Per-obstacle data packed inside ComputeCb. Layout must match ObstacleCb in ComputeCb.hlsli.
__declspec(align(16)) struct ObstacleData {
	Float4x4 invTransform;  // offset   0: world-to-object transform for SDF sampling
	Float3   sdfMin;        // offset  64: object-space SDF AABB minimum corner
	float    _pad0;         // offset  76
	Float3   sdfMax;        // offset  80: object-space SDF AABB maximum corner
	float    _pad1;         // offset  92
	// 96 bytes per obstacle
};

// Per-simulation-step data uploaded to all PBF compute shaders before each dispatch.
// Fields that are compile-time constants (h, rho0, gridMin/Max, kernel coefficients,
// push radius, sCorrDeltaQ/N) have been removed and are now #defines in SharedConfig.hlsli.
// Only runtime-variable fields remain here.
// Must be 16-byte aligned; fields ordered to avoid straddling 16-byte boundaries.
__declspec(align(16)) struct ComputeCb {
	float dt;               // offset  0: simulation timestep in seconds
	UINT numParticles;      // offset  4: total particle count; bounds check in every shader
	float sCorrK;           // offset  8: artificial pressure magnitude coefficient
	float vorticityEpsilon; // offset 12: vorticity confinement strength
	Float3 boxMin;          // offset 16: simulation box minimum corner (adjustable, world space)
	float epsilon;          // offset 28: constraint force mixing relaxation
	Float3 boxMax;          // offset 32: simulation box maximum corner (adjustable, world space)
	float viscosity;        // offset 44: XSPH viscosity coefficient
	Float3 externalForce;   // offset 48: horizontal acceleration from arrow keys (m/s^2)
	UINT fountainEnabled;   // offset 60: 1 = fountain jet active, 0 = off
	float adhesion;         // offset 64: tangential velocity damping on wall contact
	float viewportWidth;    // offset 68: render target width in pixels
	float viewportHeight;   // offset 72: render target height in pixels
	float pushRadius;       // offset 76: solid push-out distance; PUSH_RADIUS in particle modes, 0 in liquid mode
	UINT minLOD;            // offset 80: minimum solver iterations (farthest particles)
	UINT maxLOD;            // offset 84: maximum solver iterations (= solverIterations, closest)
	float sbdSuctionStrength; // offset 88: position-correction strength of SBD nodes on nearby fluid
	float _padB;            // offset 92
	Float3 cameraPos;       // offset 96: camera world position for DTC LOD computation
	float _padC;            // offset 108
	Float4x4 viewProjTransform; // offset 112: world-to-clip transform for DTVS depth projection
	// offset 176: obstacles array placed last so extending NUM_OBSTACLES only grows the tail
	ObstacleData obstacles[NUM_OBSTACLES]; // offset 176: per-obstacle transforms and SDF bounds
};
