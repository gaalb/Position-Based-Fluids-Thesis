// Soft body dynamics: apply strain-based constraints to predicted positions.
// One pass per solver iteration; corrects predictedPosition in place.
// Input/Output: predictedPosition (u0)
// orion (b1, root constant): encodes orientation and parity for this dispatch.
//   bits 0-1: axes.x (main edge axis, 0=X 1=Y 2=Z)
//   bits 2-3: axes.y (second neighbor axis)
//   bits 4-5: axes.z (third neighbor axis)
//   bit  6:   signyz.x (1=positive, 0=negative direction for second neighbor)
//   bit  7:   signyz.y (1=positive, 0=negative direction for third neighbor)
//   bit  8:   parity (0 or 1, selects even/odd A-node set to avoid write conflicts)
#define SbdStrainRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors=2)), RootConstants(num32BitConstants=1, b1)"

cbuffer SbdOrionCb : register(b1) { uint orion; };

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);

#include "ComputeCb.hlsli"

float squaredLength(float3 v) {
	return dot(v, v);
}

float3x3 outerProduct(float3 a, float3 b)
{
    return float3x3(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    );
}

float3x3 get_nabla_p_Sij(const float3x3 F, const float3x3 Qi, uint i, uint j) {
	return float3x3(
		outerProduct(Qi[i], F[j]) + outerProduct(Qi[j], F[i])
	);
}

float3x3 get_overline_nabla_p_Sij(const float3x3 F, const float3x3 Qi, float Sij, uint i, uint j) {
	float fi2 = dot(F[i], F[i]) + 1e-3;
    float fili = rsqrt(fi2 + 1e-3);
	float fj2 = dot(F[j], F[j]) + 1e-3;
    float fjli = rsqrt(fj2 + 1e-3);

    float3x3 result = get_nabla_p_Sij(F, Qi, i, j);
   // result -= (outerProduct(Qi[i], F[j]) * fj2 + outerProduct(Qi[j], F[i]) * fi2) * Sij / fi2 / fj2;
//	result -= (outerProduct(Qi[i], F[j]) / fi2 + outerProduct(Qi[j], F[i]) / fj2 ) * Sij;	
	return result * fili * fjli;
}

float get_lambda_stretch(float Sii, float sum_length_of_nabla_p_Sii) {
	float sqrtSii = sqrt(Sii);
	return 2.0f * sqrtSii * (sqrtSii - 1.0f) / (sum_length_of_nabla_p_Sii + 1e-3);
}

float get_lambda_shear(float Sij, float sum_length_of_nabla_p_Sij) {
	return Sij / (sum_length_of_nabla_p_Sij + 1e-3);
}

float get_lambda_volume(float Cvol, float sum_length_of_nabla_Cvol) {
	return Cvol / (sum_length_of_nabla_Cvol + 1e-3);
}

float3x3 get_delta_p_for_stretch(const float3x3 F, const float3x3 Qi, const float Sii, uint i) {
	float3x3 nabla_p_Sii = get_nabla_p_Sij(F, Qi, i, i);

	float sum_length =
		squaredLength(nabla_p_Sii[0]) +
		squaredLength(nabla_p_Sii[1]) +
		squaredLength(nabla_p_Sii[2]) +
		squaredLength(nabla_p_Sii[0] + nabla_p_Sii[1] + nabla_p_Sii[2])
	;

	return nabla_p_Sii * -get_lambda_stretch(Sii, sum_length);
}

float3x3 get_delta_p_for_stretch(const float3x3 F, const float3x3 Qi, const float3x3 S) {
    return 
		get_delta_p_for_stretch(F, Qi, S[0][0], 0) + 
		get_delta_p_for_stretch(F, Qi, S[1][1], 1) + 
		get_delta_p_for_stretch(F, Qi, S[2][2], 2) 
	;
}

float3x3 get_delta_p_for_shear(const float3x3 F, const float3x3 Qi, const float Sij, uint i, uint j) {
	float3x3 nabla_p_Sij = get_overline_nabla_p_Sij(F, Qi, Sij, i, j);

	float sum_length =
		squaredLength(nabla_p_Sij[0]) +
		squaredLength(nabla_p_Sij[1]) +
		squaredLength(nabla_p_Sij[2]) +
		squaredLength(nabla_p_Sij[0] + nabla_p_Sij[1] + nabla_p_Sij[2])
	;

	return nabla_p_Sij * -get_lambda_shear(Sij, sum_length);
	
}

float3x3 get_delta_p_for_shear(const float3x3 F, const float3x3 Qi, const float3x3 S) {
	return
		get_delta_p_for_shear(F, Qi, S[1][0], 1, 0) +
		get_delta_p_for_shear(F, Qi, S[2][0], 2, 0) +
		get_delta_p_for_shear(F, Qi, S[2][1], 2, 1)
	;
}

float3x3 get_delta_p_for_volume(const float3x3 P, const float3x3 Q) {
	//nabla p only at first
	float3 dpd1 = cross(P[1], P[2]);
	float3 dpd2 = cross(P[2], P[0]);
	float3 dpd3 = cross(P[0], P[1]);

	float sum_length = 
		squaredLength(dpd1) +
		squaredLength(dpd2) +
		squaredLength(dpd3) +
		squaredLength(dpd1 + dpd2 + dpd3)
	;

	float Cvol = dot (Q[0], cross(Q[1], Q[2])) - dot (P[0], dpd1);
	return float3x3(
		dpd1 ,
		dpd2 ,
		dpd3
	) * Cvol / (sum_length);
}

void executeConstraintsOnVertices(float3x3 Q, float3x3 Qi, inout float3 p0, inout float3 p1, inout float3 p2, inout float3 p3) {

	float3x3 P = float3x3(
			p1 - p0,
			p2 - p0,
			p3 - p0
		);
	//if(abs(determinant(P)) < 100.0){
	//	return;			  
	//}
	
	float3x3 F = mul(Qi, P);
	float3x3 S = mul(F, transpose(F));

	//Stretch
	float3x3 deltaP = 
		get_delta_p_for_stretch(F, Qi, S) * 0.03 * 10.0
		+ 
		get_delta_p_for_shear(F, Qi, S) * 0.02 * 10.0
		+
		get_delta_p_for_volume(P, Q) * 0.03 * 10.0
	;
	
    p0 -= (deltaP[0] + deltaP[1] + deltaP[2]);
    p1 += deltaP[0];
    p2 += deltaP[1];
    p3 += deltaP[2];
}

static const int3 orientations[12] =
{
	int3( 0,  1,  2),
	int3( 0,  2, -1),
	int3( 0, -1, -2),
	int3( 0, -2,  1),
	int3( 2,  0,  1),
	int3(-1,  0,  2),
	int3(-2,  0, -1),
	int3( 1,  0, -2),
	int3( 1,  2,  0),
	int3( 2, -1,  0),
	int3(-1, -2,  0),
	int3(-2,  1,  0)
};

void clump(float3x3 Q, float3x3 Qi, inout float3 p0, inout float3 p1, inout float3 p2, inout float3 p3)
{
    float3 mid = (p0 + p1 + p2 + p3) * 0.25;
    p0 = mid + (p0 - mid) * 0.99;
    p1 = mid + (p1 - mid) * 0.99;
    p2 = mid + (p2 - mid) * 0.99;
    p3 = mid + (p3 - mid) * 0.99;
}

[RootSignature(SbdStrainRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
	uint3 axes = uint3(
//	    0, 1, 2
		(orion >> 0) & 0x3,
		(orion >> 2) & 0x3,
		(orion >> 4) & 0x3
	);
	uint3 dims = uint3(SBD_DIM_X, SBD_DIM_Y, SBD_DIM_Z);
	dims[axes.x] /= 2;
    if (tid >= dims.x * dims.y * dims.z) return;
	
    uint2 signyz = uint2((orion >> 6) & 0x1, (orion >> 7) & 0x1);
    float3 signs = float3(
//	1.0, 1.0, 1.0
		1.0,
		signyz.x ? 1.0 : -1.0,
		signyz.y ? 1.0 : -1.0
	);

//	float3x3 Q = float3x3(
//		float3(2.0, 0.0, 0.0),
//		float3(1.0, 1.0,-1.0),
//		float3(1.0, 1.0, 1.0)
//	);	
	float3x3 Q = float3x3(
		float3(0.0, 0.0, 0.0), // must be float3(2.0, 0.0,  0.0); swizzled
		float3(1.0, 1.0, 1.0), // must be float3(1.0, 1.0, -1.0); swizzled
		float3(1.0, 1.0, 1.0)
	);
	Q[0][axes.x] = 2.0;
	Q[1][axes.z] *= -1.0;
	Q[1][axes.y] *= signs[1];
	Q[2][axes.y] *= signs[1];	
	Q[1][axes.z] *= signs[2];
	Q[2][axes.z] *= signs[2];

	float3x3 Qi;
	Qi[axes.x] = float3(0.5, 0.0,  0.0);
	Qi[axes.y] = float3(-0.5, 0.5, 0.5) * signs[1];
	Qi[axes.z] = float3(0.0, -0.5, 0.5) * signs[2];


	uint3 iid;
	iid.x = tid % dims.x;
	iid.y = (tid / dims.x) % dims.y;
	iid.z = tid / dims.x / dims.y;
	iid[axes.x] *= 2;
	iid[axes.x] += (((orion >> 8) + iid[axes.y] + iid[axes.z] ) & 0x1);

    uint3 strides = uint3(1, SBD_DIM_X, SBD_DIM_X * SBD_DIM_Y);
    uint nid = iid.x + (iid.y + iid.z * SBD_DIM_Y) * SBD_DIM_X;
    iid[axes.x] += 1;
	iid[axes.y] += signyz.x;
	iid[axes.z] += 1 - signyz.y;
    uint qid = iid.x + (iid.y + iid.z * (SBD_DIM_Y+1)) * (SBD_DIM_X+1) + SBD_NUM_CUBIC_NODES;
	iid[axes.z] += 2 * signyz.y - 1;
    uint qid2 = iid.x + (iid.y + iid.z * (SBD_DIM_Y+1)) * (SBD_DIM_X+1) + SBD_NUM_CUBIC_NODES;	
	
    // TODO: compute strain from BCC neighbor distances and apply position correction
	//predictedPosition[qid].x += 0.1;
	//predictedPosition[nid + strides[axes.x]].x += 0.01;
    float3 p0 = predictedPosition[nid];
    float3 p1 = predictedPosition[nid + strides[axes.x]];
    float3 p2 = predictedPosition[qid];
    float3 p3 = predictedPosition[qid2];
	//clump(
	executeConstraintsOnVertices(
		Q,
		Qi,
		predictedPosition[nid],
		predictedPosition[nid + strides[axes.x]],
		predictedPosition[qid],
		predictedPosition[qid2]
	);

	// Velocity damping: eq. 23 from the SBD paper.
	// n_i = dp_i / |dp_i|, alpha = sum_j(v_j . n_j), v_i -= k * alpha * n_i
	// NOTE: currently disabled. updateVelocityCS overwrites velocity with (pred-pos)/dt
	// after all strain passes, so any writes here are discarded. The correct fix is to
	// apply this formula in a second constraint pass after updateVelocityCS.
//	float3 dp0 = predictedPosition[nid]                   - p0;
//	float3 dp1 = predictedPosition[nid + strides[axes.x]] - p1;
//	float3 dp2 = predictedPosition[qid]                   - p2;
//	float3 dp3 = predictedPosition[qid2]                  - p3;
//
//	float l0 = length(dp0), l1 = length(dp1), l2 = length(dp2), l3 = length(dp3);
//	float3 n0 = l0 > 1e-8f ? dp0 / l0 : (float3)0;
//	float3 n1 = l1 > 1e-8f ? dp1 / l1 : (float3)0;
//	float3 n2 = l2 > 1e-8f ? dp2 / l2 : (float3)0;
//	float3 n3 = l3 > 1e-8f ? dp3 / l3 : (float3)0;
//
//	float3 v0 = velocity[nid];
//	float3 v1 = velocity[nid + strides[axes.x]];
//	float3 v2 = velocity[qid];
//	float3 v3 = velocity[qid2];
//
//	float alpha = dot(v0, n0) + dot(v1, n1) + dot(v2, n2) + dot(v3, n3);
//	float k = /* sbdDampingK */ 0.5f * alpha;  // sbdDampingK was removed; replace with a dedicated CB field when re-enabling
//
//	velocity[nid]                   = v0 - k * n0;
//	velocity[nid + strides[axes.x]] = v1 - k * n1;
//	velocity[qid]                   = v2 - k * n2;
//	velocity[qid2]                  = v3 - k * n3;
/*	
	float3x3 deltaP = executeConstraintsOnVertices(
		Q,
		Qi,
		p0,
		p1,
		p2,
		p3
	);
    predictedPosition[nid] = p0 - deltaP[0] - deltaP[1] - deltaP[2];
    predictedPosition[nid + strides[axes.x]] = p1 + deltaP[0];
    predictedPosition[qid] = p2 + deltaP[1];
    predictedPosition[qid2] = p3 + deltaP[2];
*/
}
