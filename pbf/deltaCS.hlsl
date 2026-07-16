// For each particle i, compute a position correction:
// delta_p_i = (1/rho0) * sum_{j != i}( (lambda_i + lambda_j) * grad_W_spiky(r_ij, h) )
//
// Apply the correction to the predicted position:
// p*_i += delta_p_i
//
// To avoid a Gauss-Seidel race (thread i reading predictedPosition[j] while thread j
// overwrites it in the same dispatch), this shader writes its result to the scratch field
// scratch. A separate positionFromScratchCS pass then copies the scratch field back to
// predictedPosition before the next solver iteration, ensuring all threads in the next
// iteration see a consistent snapshot.
//
// In: predictedPosition, lambda, cellCount, cellPrefixSum, lod
// Out: scratch (new predicted position)

#define DeltaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 6))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli" // SpikyGrad, Poly6
#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<float>  lambda            : register(u1);
RWStructuredBuffer<float3> scratch           : register(u2);
RWStructuredBuffer<uint>   cellCount         : register(u3);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u4);
RWStructuredBuffer<uint>   lod               : register(u5);

[RootSignature(DeltaRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) 
        return; // skip threads that belong to no particle
    if (lod[i] == 0)
        return; // skip particles that have exhausted their iteration count

    // cache to avoid repeated UAV reads
    float3 pi = predictedPosition[i];
    float lambdaI = lambda[i];

    // Precompute the s_corr denominator, same for every (i,j) pair.
    // Poly6 only uses the squared magnitude so the direction does not matter.
    float poly6AtDeltaQ = Poly6(float3(SCORR_DELTA_Q, 0, 0), SCORR_DELTA_Q * SCORR_DELTA_Q);

    float3 deltaP = float3(0, 0, 0);

    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];
        uint prefixSum = cellPrefixSum[ci];
        
        for (uint s = 0; s < count; s++)
        {
            uint j = prefixSum + s;
            if (j == i)
                continue;

            // r_ij points from neighbor j toward particle i
            float3 r = pi - predictedPosition[j];
            
            float r2 = dot(r, r);

            // Overlapping particles (r ~ 0): SpikyGrad returns zero so they'd
            // be stuck forever. Skip the normal sCorr + SpikyGrad computation
            // (which would blow up due to Poly6(0)/Poly6(deltaQ) in sCorr) and
            // instead add a small direct repulsive nudge in a pseudo-random
            // direction. On subsequent solver iterations, even a tiny separation
            // lets the normal gradient take over.
             if (r2 < EPSILON*EPSILON) {
                deltaP += overlapJitter(i, j) * (H * 0.001);
                continue;
            }

            // artificial pressure term s_corr to suppress tensile instability.
            // When lambda > 0 (sparse region), the standard Eq. 12 correction becomes attractive,
            // pulling surface particles into tight clumps. s_corr adds a small repulsive bias
            // that counteracts this without disturbing the bulk behavior.
            // s_corr = -k * (W(r, h) / W(delta_q, h))^n
            float wRatio = Poly6(r, r2) / poly6AtDeltaQ;
            float sCorr = -sCorrK * pow(wRatio, SCORR_N); // sCorrK > 0, pow >= 0, so sCorr <= 0 always

            // position correction with artificial pressure included.
            // SpikyGrad with r = p_i - p_j points from i toward j.
            // A negative coefficient times that direction pushes i away from j -- repulsive.
            // So sCorr is a repulsive contribution. The "surface tension-like effect" is because
            // sCorr keeps the bulk density slightly below rho0, so even bulk particles
            // end up with a weakly positive lambda (attractive). At the surface, where
            // particle counts are low and density is even lower, this mild attraction
            // creates a coherent surface instead of the violent clumping (tensile instability)
            // that occurs without s_corr. Raising sCorrK increases this surface cohesion.
            deltaP += (lambdaI + lambda[j] + sCorr) * SpikyGrad(r, r2);
        }
    }
    deltaP /= RHO0;
    //if(pi.x < 0.0 ){
    //    deltaP *= 1;
    //}

    // Update the predicted position (collision response is handled by collisionCS)
    scratch[i] = pi + deltaP;
}
