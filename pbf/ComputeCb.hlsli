#ifndef COMPUTE_CB_HLSLI
#define COMPUTE_CB_HLSLI

#include "SharedConfig.hlsli"

// Per-simulation-step data uploaded to all PBF compute shaders before each dispatch.
// Fields that are compile-time constants (h, rho0, gridMin/Max, kernel coefficients,
// push radius, sCorrDeltaQ/N) have been removed and are now #defines in SharedConfig.hlsli.
// Only runtime-variable fields remain here.

// Per-obstacle data packed inside ComputeCb. Layout matches C++ ObstacleData.
struct ObstacleCb {
    float4x4 invTransform; // offset   0: world-to-object transform for SDF sampling
    float3   sdfMin;       // offset  64: object-space SDF AABB minimum corner
    float    _pad0;        // offset  76
    float3   sdfMax;       // offset  80: object-space SDF AABB maximum corner
    float    _pad1;        // offset  92
    // 96 bytes per obstacle
};

cbuffer ComputeCb : register(b0)
{
    float dt;               // offset  0: simulation timestep in seconds
    uint numParticles;      // offset  4: total particle count; bounds check in every shader
    float sCorrK;           // offset  8: artificial pressure magnitude coefficient
    float vorticityEpsilon; // offset 12: vorticity confinement strength
    float3 boxMin;          // offset 16: simulation box minimum corner (adjustable, world space)
    float epsilon;          // offset 28: constraint force mixing relaxation
    float3 boxMax;          // offset 32: simulation box maximum corner (adjustable, world space)
    float viscosity;        // offset 44: XSPH viscosity coefficient
    float3 externalForce;   // offset 48: horizontal acceleration from arrow keys (m/s^2)
    uint fountainEnabled;   // offset 60: 1 = fountain jet active, 0 = off
    float adhesion;         // offset 64: tangential velocity damping on wall contact
    float viewportWidth;    // offset 68: render target width in pixels
    float viewportHeight;   // offset 72: render target height in pixels
    float pushRadius;       // offset 76: solid push-out distance; PUSH_RADIUS normally, 0 in liquid mode
    uint minLOD;            // offset 80: minimum solver iterations (farthest particles)
    uint maxLOD;            // offset 84: maximum solver iterations (= solverIterations, closest)
    float sbdSuctionStrength; // offset 88: position-correction strength of SBD nodes on nearby fluid
    float sbdWallStrength;  // offset 92: amplitude of exponential wall-repulsion force (m/s²)
    float3 cameraPos;       // offset 96: camera world position for DTC LOD computation
    float sbdWallFalloff;   // offset 108: characteristic decay distance for wall repulsion (m)
    float4x4 viewProjTransform; // offset 112: world-to-clip transform for DTVS depth projection
    // offset 176: obstacles array placed last so extending NUM_OBSTACLES only grows the tail
    ObstacleCb obstacles[NUM_OBSTACLES]; // offset 176: per-obstacle transforms and SDF bounds
    float sbdDragStrength;      // offset 272: viscous drag pulling fluid velocity toward nearby SBD node velocities
    float charSuctionStrength;  // offset 276: pull fluid predicted positions into character volume (deficit-weighted)
    float charVelocityStrength; // offset 280: XSPH-style velocity matching of fluid to character animation
};

#endif
