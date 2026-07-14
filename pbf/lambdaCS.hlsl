// For each particle i, this shader:
// Estimates local density rho_i by summing Poly6 kernel contributions from all j:
// rho_i = sum_j(W_poly6(p_i - p_j, h))  (m = 1)
//
// Evaluates the density constraint: C_i = rho_i / rho0 - 1
//
// Computes lambda_i used by deltaCS for position corrections:
// lambda_i = -C_i / ( sum_k(|grad_pk(C_i)|^2) + eps )
//
// The denominator sums the squared gradient of C_i with respect to every particle k.
// two cases depending on whether k == i or k == j:
//   k == i: grad_pi(C_i) = (1/rho0) * sum_{j != i}( grad_W_spiky(r_ij, h) )
//   k == j: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
// eps prevents division by zero when a particle has no neighbors.
//
// In: predictedPosition, cellCount, cellPreficSum, lod
// Out: lambda, density
#define LambdaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 6))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli" // Poly6, SpikyGrad
#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<float>  lambda            : register(u1);
RWStructuredBuffer<float>  density           : register(u2);
RWStructuredBuffer<uint>   cellCount         : register(u3);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u4);
RWStructuredBuffer<uint>   lod               : register(u5);

[RootSignature(LambdaRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;
    if (lod[i] == 0)
        return;

    // The lambda calculation involves three "indexes", i, j, and k, which aren't clearly explained
    // in the original research article:
    // i is the index of the particle we're computing lambda for (the "self" particle)
    // j is the index of a neighbor particle that contributes to i's density and constraint gradient
    // k is the index over a set of particles that includes the self particle and all the neighbors,
    // therefore it has two cases: k=i and k=j

    float3 pi = predictedPosition[i]; // cache to avoid repeated UAV reads

    float rho = 0.0; // density estimate rho_i
    float3 gradI = float3(0, 0, 0); // accumulates sum_{j != i}( grad_W(r_ij) ) for the k=i case
    float gradSqSum = 0.0; // accumulates sum_k(|grad_pk(C_i)|^2) for the k=j case

    // Iterate over neighboring cells using the precomputed cell index list.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];
        int prefixSum = cellPrefixSum[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = prefixSum + s;

            // r points from neighbor j toward particle i (r_ij = p_i - p_j)
            float3 ppj = predictedPosition[j];
            float3 r = pi - ppj;
            float rhosat = RHO0; // rest density
            if (ppj.x < 0.0)
                rhosat *= 1.0; // reduce rest density for particles on the left side of the box
            
            float r2 = dot(r, r);

            // Density: every particle j including i itself contributes.
            rho += Poly6(r, r2);

            if (j != i)
            {
                // k=j case: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
                float3 gradW = SpikyGrad(r, r2);
                float3 gradJ = -(1.0 / rhosat) * gradW;
                gradSqSum += dot(gradJ, gradJ); // add |grad_pj(C_i)|^2 to denominator

                // Also accumulate gradW into gradI -- needed for the k=i term after the loop
                gradI += gradW / rhosat;
            }
        }
    }

    // k=i case
    // grad_pi(C_i) = (1/rho0) * sum_{j != i}(grad_W_spiky(r_ij, h))
    // gradI now holds the raw sum; apply the 1/rho0 factor and add its squared magnitude.
//sat    gradI /= rhosat;
    gradSqSum += dot(gradI, gradI); // add |grad_pi(C_i)|^2 to denominator

    float rhosat = RHO0; // rest density
    if (pi.x < 0.0)
        rhosat *= 1.0; // reduce rest density for particles on the left side of the box
    
    // Density constraint value
    float C = rho /rhosat - 1.0; // 0 at rest density, > 0 if compressed, < 0 if sparse

    // Store density for visualization (read by rendering shaders)
    density[i] = rho;

    // Newton step length lambda
    lambda[i] = -C / (gradSqSum + epsilon);
}
