#pragma once

#include "Egg/Common.h"
#include "Egg/Mesh/Geometry.h"
#include "Egg/Shader.h"
#include "Egg/Scene/PerObjectData.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <DirectXMath.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>

using namespace DirectX;

// 128-bone skinning CB. sizeof = 128*64 = 8192 = 32*256 — 256-byte stride-aligned.
__declspec(align(16)) struct SkinCb {
    XMFLOAT4X4 bones[128];
};
static_assert(sizeof(SkinCb) % 256 == 0, "SkinCb must be 256-byte stride-aligned");

class HipHopAnimation {
    static const int MAX_BONES   = 128;
    static const int N_INSTANCES = 1;

    // ---- skinned vertex (stride = 64 bytes) ----
    struct SkinnedVertex {
        float    pos[3];
        float    norm[3];
        float    tangent[3];
        float    uv[2];
        uint8_t  boneIdx[4];
        float    boneWt[4];
    };

    // ---- skeleton ----
    struct NodeInfo {
        std::string        name;
        XMFLOAT4X4         defaultTransform;
        int                parentIdx;
        std::vector<int>   children;
        int                animChannelIdx;
    };
    struct BoneInfo {
        XMFLOAT4X4 offsetMatrix;
        int        nodeIdx;
    };

    // ---- character volume particles ----
    struct CharParticle {
        XMFLOAT3 bindPos;    // rest-pose position (mesh/bind space)
        uint8_t  boneIdx[4]; // up to 4 bone influences
        float    boneWt[4];  // corresponding normalized weights
    };
    std::vector<CharParticle> charParticles;
    std::vector<XMFLOAT3>     prevCharPos; // previous frame skinned positions (for velocity)
    com_ptr<ID3D12Resource>   charPosUpload;
    com_ptr<ID3D12Resource>   charVelUpload;
    XMFLOAT3*                 charPosMapped = nullptr;
    XMFLOAT3*                 charVelMapped = nullptr;

    // Sample nParticles interior points from the mesh and fill charParticles.
    // Uses triangle+centroid tetrahedra sampling for uniform volume distribution.
    void sampleParticlesFromMesh(
        const std::vector<SkinnedVertex>& verts,
        const std::vector<unsigned int>& indices,
        int nParticles)
    {
        // Centroid of all mesh vertices (bind space)
        XMVECTOR centroid = XMVectorZero();
        for (auto& v : verts) {
            XMFLOAT3 p(v.pos[0], v.pos[1], v.pos[2]);
            centroid = XMVectorAdd(centroid, XMLoadFloat3(&p));
        }
        centroid = XMVectorScale(centroid, 1.0f / (float)verts.size());

        UINT nTris = (UINT)indices.size() / 3;
        std::mt19937 rng(42);
        std::uniform_int_distribution<UINT> triDist(0, nTris - 1);
        std::uniform_real_distribution<float> fd(0.0f, 1.0f);

        charParticles.clear();
        charParticles.reserve(nParticles);
        prevCharPos.assign(nParticles, XMFLOAT3(0, 0, 0));

        for (int i = 0; i < nParticles; i++) {
            UINT tri = triDist(rng);
            UINT i0 = indices[tri*3], i1 = indices[tri*3+1], i2 = indices[tri*3+2];
            const SkinnedVertex& v0 = verts[i0];
            const SkinnedVertex& v1 = verts[i1];
            const SkinnedVertex& v2 = verts[i2];

            // Uniform random point on triangle (Osada et al. 2002)
            float r1 = sqrtf(fd(rng)), r2 = fd(rng);
            float a = 1.0f - r1, b = r1*(1.0f-r2), c = r1*r2;
            XMFLOAT3 triPt(
                a*v0.pos[0]+b*v1.pos[0]+c*v2.pos[0],
                a*v0.pos[1]+b*v1.pos[1]+c*v2.pos[1],
                a*v0.pos[2]+b*v1.pos[2]+c*v2.pos[2]);

            // Lerp toward centroid by random t for interior distribution
            float t = fd(rng);
            XMVECTOR finalPosV = XMVectorLerp(XMLoadFloat3(&triPt), centroid, t);

            CharParticle cp;
            XMStoreFloat3(&cp.bindPos, finalPosV);

            // Blend bone weights from the three triangle vertices (using barycentric a,b,c)
            float totalWt[128] = {};
            auto addVert = [&](const SkinnedVertex& vv, float bary) {
                for (int k = 0; k < 4; k++) totalWt[vv.boneIdx[k]] += bary * vv.boneWt[k];
            };
            addVert(v0, a); addVert(v1, b); addVert(v2, c);

            // Keep top 4 contributing bones
            struct BW { int idx; float wt; } top4[4] = {};
            for (int bi = 0; bi < 128; bi++) {
                if (totalWt[bi] > top4[3].wt) {
                    top4[3] = {bi, totalWt[bi]};
                    for (int k = 3; k > 0 && top4[k].wt > top4[k-1].wt; k--)
                        std::swap(top4[k], top4[k-1]);
                }
            }
            float wsum = top4[0].wt + top4[1].wt + top4[2].wt + top4[3].wt;
            if (wsum < 1e-6f) wsum = 1.0f;
            for (int k = 0; k < 4; k++) {
                cp.boneIdx[k] = (uint8_t)top4[k].idx;
                cp.boneWt[k]  = top4[k].wt / wsum;
            }
            charParticles.push_back(cp);
        }
    }

    // ---- animation ----
    struct Channel {
        std::vector<double>    posTimes,  rotTimes,  scaleTimes;
        std::vector<XMFLOAT3>  posKeys,   scaleKeys;
        std::vector<XMFLOAT4>  rotKeys;   // quaternion xyzw
    };

    std::vector<NodeInfo>  nodes;
    std::unordered_map<std::string,int> nodeNameToIdx;
    std::vector<BoneInfo>  boneInfos;
    std::vector<Channel>   channels;
    std::vector<XMMATRIX>  globalTransforms;
    double  animDuration       = 1.0;
    double  animTicksPerSec    = 25.0;
    XMFLOAT4X4 globalInvF;
    int     nBones             = 0;

    // ---- GPU ----
    Egg::Mesh::IndexedGeometry::P geometry;
    com_ptr<ID3D12Resource>       skinCbBuffer;
    uint8_t*                      skinCbMapped = nullptr;

    // ---- pipeline ----
    com_ptr<ID3D12RootSignature>  rootSigSkinned;
    com_ptr<ID3D12RootSignature>  rootSigSkinnedCollect;
    com_ptr<ID3D12PipelineState>  pso[4];   // indexed by mien

    // ---- instances ----
    struct Instance { XMFLOAT3 pos; float timeOff; };
    std::vector<Instance> instances;
    int baseObjectIndex = 0;

    // ---- Assimp <-> DirectX ----
    // Assimp stores matrices row-major with column vectors (translation in last column).
    // Transposing converts to DirectX row-vector convention (translation in last row).
    static XMMATRIX aiToXM(const aiMatrix4x4& m) {
        return XMMATRIX(
            m.a1, m.b1, m.c1, m.d1,
            m.a2, m.b2, m.c2, m.d2,
            m.a3, m.b3, m.c3, m.d3,
            m.a4, m.b4, m.c4, m.d4);
    }

    // ---- keyframe interpolation ----
    static XMFLOAT3 lerpVec3(const std::vector<double>& t, const std::vector<XMFLOAT3>& k, double time) {
        if (k.size() == 1) return k[0];
        int i = 0, n = (int)t.size() - 1;
        while (i < n - 1 && t[i + 1] <= time) ++i;
        if (i + 1 >= (int)t.size()) return k.back();
        float f = (t[i + 1] > t[i]) ? (float)((time - t[i]) / (t[i + 1] - t[i])) : 0.f;
        return { k[i].x + (k[i+1].x - k[i].x)*f,
                 k[i].y + (k[i+1].y - k[i].y)*f,
                 k[i].z + (k[i+1].z - k[i].z)*f };
    }
    static XMVECTOR slerpQuat(const std::vector<double>& t, const std::vector<XMFLOAT4>& k, double time) {
        if (k.size() == 1) return XMLoadFloat4(&k[0]);
        int i = 0, n = (int)t.size() - 1;
        while (i < n - 1 && t[i + 1] <= time) ++i;
        if (i + 1 >= (int)t.size()) return XMLoadFloat4(&k.back());
        float f = (t[i + 1] > t[i]) ? (float)((time - t[i]) / (t[i + 1] - t[i])) : 0.f;
        return XMQuaternionSlerp(XMLoadFloat4(&k[i]), XMLoadFloat4(&k[i+1]), f);
    }

    // ---- node tree ----
    void buildNodeTree(const aiNode* node, int parentIdx) {
        int idx = (int)nodes.size();
        nodes.push_back({});
        nodes[idx].name             = node->mName.C_Str();
        nodes[idx].parentIdx        = parentIdx;
        nodes[idx].animChannelIdx   = -1;
        XMStoreFloat4x4(&nodes[idx].defaultTransform, aiToXM(node->mTransformation));
        nodeNameToIdx[nodes[idx].name] = idx;
        for (unsigned c = 0; c < node->mNumChildren; ++c) {
            int childIdx = (int)nodes.size();
            nodes[idx].children.push_back(childIdx);
            buildNodeTree(node->mChildren[c], idx);
        }
    }

    void computeGlobal(int nodeIdx, const XMMATRIX& parentGlobal, double ticks) {
        const NodeInfo& ni = nodes[nodeIdx];
        XMMATRIX local;
        if (ni.animChannelIdx >= 0) {
            const Channel& ch = channels[ni.animChannelIdx];
            XMFLOAT3 pos   = lerpVec3(ch.posTimes,   ch.posKeys,   ticks);
            XMFLOAT3 scale = lerpVec3(ch.scaleTimes,  ch.scaleKeys, ticks);
            XMVECTOR rot   = slerpQuat(ch.rotTimes,   ch.rotKeys,   ticks);
            local = XMMatrixScaling(scale.x, scale.y, scale.z)
                  * XMMatrixRotationQuaternion(rot)
                  * XMMatrixTranslation(pos.x, pos.y, pos.z);
        } else {
            local = XMLoadFloat4x4(&ni.defaultTransform);
        }
        globalTransforms[nodeIdx] = local * parentGlobal;
        for (int child : ni.children)
            computeGlobal(child, globalTransforms[nodeIdx], ticks);
    }

    // ---- PSO helper ----
    com_ptr<ID3D12PipelineState> makePso(
        ID3D12Device* dev, ID3D12RootSignature* rs,
        ID3DBlob* vs, ID3DBlob* ps,
        const D3D12_INPUT_LAYOUT_DESC& il,
        bool depthTest, D3D12_COMPARISON_FUNC depthFunc,
        int numRT, DXGI_FORMAT rtFmt = DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
        d.pRootSignature       = rs;
        d.VS                   = CD3DX12_SHADER_BYTECODE(vs);
        d.PS                   = CD3DX12_SHADER_BYTECODE(ps);
        d.InputLayout          = il;
        d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        d.RasterizerState      = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        d.BlendState           = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        d.SampleMask           = UINT_MAX;
        d.SampleDesc.Count     = 1;
        d.NumRenderTargets     = numRT;
        if (numRT > 0) d.RTVFormats[0] = rtFmt;
        auto& ds = d.DepthStencilState;
        ds = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        ds.DepthEnable    = depthTest ? TRUE : FALSE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        ds.DepthFunc      = depthFunc;
        d.DSVFormat = depthTest ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
        com_ptr<ID3D12PipelineState> p;
        DX_API("create skinned PSO") dev->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(p.GetAddressOf()));
        return p;
    }

public:
    void createResources(ID3D12Device* device, int objectSlotBase) {
        baseObjectIndex = objectSlotBase;

        instances.resize(N_INSTANCES);
        for (int i = 0; i < N_INSTANCES; ++i) {
            instances[i].pos     = { -24.0f , -10.0f, 30.0f + i * 20.0f };
            instances[i].timeOff = i * 0.5f;
        }

        // SkinCb upload buffer for all instances
        UINT64 skinBufSize = (UINT64)N_INSTANCES * sizeof(SkinCb);
        DX_API("create SkinCb buffer")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(skinBufSize),
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(skinCbBuffer.GetAddressOf()));
        skinCbBuffer->SetName(L"SkinCb");
        CD3DX12_RANGE rr(0, 0);
        skinCbBuffer->Map(0, &rr, reinterpret_cast<void**>(&skinCbMapped));

        // Load FBX
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile("../Media/kachu.fbx",
            aiProcess_Triangulate | aiProcess_LimitBoneWeights |
            aiProcess_JoinIdenticalVertices | aiProcess_SortByPType |
            aiProcess_CalcTangentSpace);
        ASSERT(scene && scene->mNumMeshes > 0, "HipHopAnimation: failed to load kachu.fbx");
        // Global inverse (cancels root transform so skinned verts stay in mesh space)
        XMStoreFloat4x4(&globalInvF,
            XMMatrixInverse(nullptr, aiToXM(scene->mRootNode->mTransformation)));

        // Build node tree
        nodes.reserve(256);
        buildNodeTree(scene->mRootNode, -1);
        globalTransforms.resize(nodes.size());

        // Collect bones from first mesh
        const aiMesh* mesh = scene->mMeshes[0];
        nBones = (int)mesh->mNumBones;
        ASSERT(nBones <= MAX_BONES, "kachu.fbx exceeds MAX_BONES limit");
        boneInfos.resize(nBones);

        // Per-vertex weight accumulation
        std::vector<std::vector<std::pair<int,float>>> vertWeights(mesh->mNumVertices);
        for (int b = 0; b < nBones; ++b) {
            const aiBone* ab = mesh->mBones[b];
            std::string boneName = ab->mName.C_Str();
            XMStoreFloat4x4(&boneInfos[b].offsetMatrix, aiToXM(ab->mOffsetMatrix));
            auto it = nodeNameToIdx.find(boneName);
            boneInfos[b].nodeIdx = (it != nodeNameToIdx.end()) ? it->second : 0;
            for (unsigned w = 0; w < ab->mNumWeights; ++w)
                vertWeights[ab->mWeights[w].mVertexId].push_back({ b, ab->mWeights[w].mWeight });
        }

        // Build vertex buffer
        std::vector<SkinnedVertex> verts(mesh->mNumVertices);
        for (unsigned vi = 0; vi < mesh->mNumVertices; ++vi) {
            auto& sv = verts[vi];
            sv.pos[0]  = mesh->mVertices[vi].x;
            sv.pos[1]  = mesh->mVertices[vi].y;
            sv.pos[2]  = mesh->mVertices[vi].z;
            sv.norm[0]    = mesh->mNormals   ? mesh->mNormals[vi].x   : 0.f;
            sv.norm[1]    = mesh->mNormals   ? mesh->mNormals[vi].y   : 0.f;
            sv.norm[2]    = mesh->mNormals   ? mesh->mNormals[vi].z   : 1.f;
            sv.tangent[0] = mesh->mTangents  ? mesh->mTangents[vi].x  : 1.f;
            sv.tangent[1] = mesh->mTangents  ? mesh->mTangents[vi].y  : 0.f;
            sv.tangent[2] = mesh->mTangents  ? mesh->mTangents[vi].z  : 0.f;
            sv.uv[0]      = (mesh->mTextureCoords[0]) ? mesh->mTextureCoords[0][vi].x : 0.f;
            sv.uv[1]      = (mesh->mTextureCoords[0]) ? mesh->mTextureCoords[0][vi].y : 0.f;
            auto& wl = vertWeights[vi];
            std::sort(wl.begin(), wl.end(), [](auto& a, auto& b){ return a.second > b.second; });
            if (wl.size() > 4) wl.resize(4);
            float wsum = 0.f; for (auto& p : wl) wsum += p.second;
            if (wsum < 1e-6f) wsum = 1.f;
            for (int k = 0; k < 4; ++k) {
                sv.boneIdx[k] = (k < (int)wl.size()) ? (uint8_t)wl[k].first : 0;
                sv.boneWt[k]  = (k < (int)wl.size()) ? wl[k].second / wsum  : 0.f;
            }
        }

        // Build index buffer
        std::vector<unsigned int> indices;
        indices.reserve(mesh->mNumFaces * 3);
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }

        geometry = Egg::Mesh::IndexedGeometry::Create(device,
            verts.data(),   (unsigned)(verts.size()   * sizeof(SkinnedVertex)), sizeof(SkinnedVertex),
            indices.data(), (unsigned)(indices.size()  * sizeof(unsigned int)));

        static const D3D12_INPUT_ELEMENT_DESC inputElems[] = {
            {"POSITION",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL",       0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD",     0, DXGI_FORMAT_R32G32_FLOAT,       0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT,      0, 44, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"BLENDWEIGHT",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 48, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        for (auto& e : inputElems) geometry->AddInputElement(e);

        // Animation channels
        if (scene->mNumAnimations > 0) {
            const aiAnimation* anim = scene->mAnimations[0];
            animDuration    = anim->mDuration;
            animTicksPerSec = (anim->mTicksPerSecond > 1e-9) ? anim->mTicksPerSecond : 25.0;
            channels.resize(anim->mNumChannels);
            for (unsigned c = 0; c < anim->mNumChannels; ++c) {
                const aiNodeAnim* na = anim->mChannels[c];
                auto it = nodeNameToIdx.find(na->mNodeName.C_Str());
                if (it != nodeNameToIdx.end())
                    nodes[it->second].animChannelIdx = (int)c;
                Channel& ch = channels[c];
                for (unsigned k = 0; k < na->mNumPositionKeys; ++k) {
                    ch.posTimes.push_back(na->mPositionKeys[k].mTime);
                    auto& v = na->mPositionKeys[k].mValue;
                    ch.posKeys.push_back({ v.x, v.y, v.z });
                }
                for (unsigned k = 0; k < na->mNumRotationKeys; ++k) {
                    ch.rotTimes.push_back(na->mRotationKeys[k].mTime);
                    auto& q = na->mRotationKeys[k].mValue;
                    ch.rotKeys.push_back({ q.x, q.y, q.z, q.w });
                }
                for (unsigned k = 0; k < na->mNumScalingKeys; ++k) {
                    ch.scaleTimes.push_back(na->mScalingKeys[k].mTime);
                    auto& v = na->mScalingKeys[k].mValue;
                    ch.scaleKeys.push_back({ v.x, v.y, v.z });
                }
            }
        }

        // Shaders & PSOs
        com_ptr<ID3DBlob> vsS  = Egg::Shader::LoadCso("Shaders/Retam/retamSkinnedVS.cso");
        com_ptr<ID3DBlob> vsC  = Egg::Shader::LoadCso("Shaders/Retam/retamSkinnedCollectVS.cso");
        com_ptr<ID3DBlob> ps0  = Egg::Shader::LoadCso("Shaders/Retam/retam256PS.cso");
        com_ptr<ID3DBlob> ps1  = Egg::Shader::LoadCso("Shaders/Retam/retam256SkinnedCollectPS.cso");
        com_ptr<ID3DBlob> ps2  = Egg::Shader::LoadCso("Shaders/Retam/layDownDepthPS.cso");
        com_ptr<ID3DBlob> ps3  = Egg::Shader::LoadCso("Shaders/Retam/retamReCollectPS.cso");

        rootSigSkinned        = Egg::Shader::LoadRootSignature(device, vsS.Get());
        rootSigSkinnedCollect = Egg::Shader::LoadRootSignature(device, vsC.Get());

        D3D12_INPUT_LAYOUT_DESC il = geometry->GetInputLayout();

        pso[0] = makePso(device, rootSigSkinned.Get(),        vsS.Get(), ps0.Get(), il, true,  D3D12_COMPARISON_FUNC_LESS,          1, DXGI_FORMAT_R8G8B8A8_UNORM);
        pso[1] = makePso(device, rootSigSkinnedCollect.Get(), vsC.Get(), ps1.Get(), il, true,  D3D12_COMPARISON_FUNC_LESS_EQUAL,    1, DXGI_FORMAT_R8G8B8A8_UNORM);
        pso[2] = makePso(device, rootSigSkinned.Get(),        vsS.Get(), ps2.Get(), il, true,  D3D12_COMPARISON_FUNC_LESS,          0);
        pso[3] = makePso(device, rootSigSkinnedCollect.Get(), vsC.Get(), ps3.Get(), il, true,  D3D12_COMPARISON_FUNC_LESS_EQUAL,    1, DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    // Called every frame from RetamApp::Update
    void update(float T, Egg::Scene::PerObjectData* objectSlots) {
        XMMATRIX globalInv = XMLoadFloat4x4(&globalInvF);

        for (int inst = 0; inst < N_INSTANCES; ++inst) {
            double ticks = fmod((T + instances[inst].timeOff) * animTicksPerSec, animDuration);
            if (ticks < 0.0) ticks += animDuration;

            computeGlobal(0, XMMatrixIdentity(), ticks);

            // Write bone matrices: offset * globalBone * globalInverse (DX row-vector order)
            SkinCb* cb = reinterpret_cast<SkinCb*>(skinCbMapped + (size_t)inst * sizeof(SkinCb));
            for (int b = 0; b < nBones; ++b) {
                XMMATRIX offset = XMLoadFloat4x4(&boneInfos[b].offsetMatrix);
                XMMATRIX global = globalTransforms[boneInfos[b].nodeIdx];
                XMStoreFloat4x4(&cb->bones[b], offset * global * globalInv);
            }

            // Model matrix for this instance (simple translation)
            XMMATRIX model    = XMMatrixTranslation(instances[inst].pos.x, instances[inst].pos.y, instances[inst].pos.z);
            XMMATRIX modelInv = XMMatrixInverse(nullptr, model);
            XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&objectSlots[inst].modelTransform),        model);
            XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&objectSlots[inst].modelTransformInverse), modelInv);
        }
    }

    // Called per-mien during PopulateCommandList
    void draw(ID3D12GraphicsCommandList* cmd, int mien,
              ID3D12DescriptorHeap* uavHeap, UINT uavDescIncr,
              D3D12_GPU_VIRTUAL_ADDRESS retamMatCbAddr,
              D3D12_GPU_VIRTUAL_ADDRESS perFrameCbAddr,
              D3D12_GPU_VIRTUAL_ADDRESS perObjectCbAddr)
    {
        if (mien < 0 || mien > 3) return;
        bool isCollect = (mien == 1 || mien == 3);

        cmd->SetPipelineState(pso[mien].Get());
        cmd->SetGraphicsRootSignature(isCollect ? rootSigSkinnedCollect.Get() : rootSigSkinned.Get());

        ID3D12DescriptorHeap* heaps[] = { uavHeap };
        cmd->SetDescriptorHeaps(1, heaps);

        // UAV table (collect passes, root param 4) — set once
        if (isCollect) {
            cmd->SetGraphicsRootDescriptorTable(4, uavHeap->GetGPUDescriptorHandleForHeapStart());
        }
        // collect: uvmask (slot 10) at root param 5; non-collect: carrot.jpg (slot 11) at root param 4
        int srvSlot = isCollect ? 10 : 11;
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(uavHeap->GetGPUDescriptorHandleForHeapStart(), srvSlot, uavDescIncr);
        cmd->SetGraphicsRootDescriptorTable(isCollect ? 5 : 4, srvHandle);

        cmd->SetGraphicsRootConstantBufferView(0, retamMatCbAddr);
        cmd->SetGraphicsRootConstantBufferView(1, perFrameCbAddr);

        for (int inst = 0; inst < N_INSTANCES; ++inst) {
            cmd->SetGraphicsRootConstantBufferView(2,
                perObjectCbAddr + (UINT64)(baseObjectIndex + inst) * sizeof(Egg::Scene::PerObjectData));
            cmd->SetGraphicsRootConstantBufferView(3,
                skinCbBuffer->GetGPUVirtualAddress() + (UINT64)inst * sizeof(SkinCb));
            geometry->Draw(cmd);
        }
    }

    int getBaseObjectIndex() const { return baseObjectIndex; }

    // ---- Character particle API (for PbfApp fluid interaction) ----

    // Load the FBX, build the skeleton and animation channels, sample N_CHAR_PARTICLES
    // interior volume points, and create persistently-mapped upload heap buffers.
    // Does NOT create rendering PSOs or upload geometry — safe to call from PbfApp
    // without Retam shader infrastructure.
    bool createParticleResources(ID3D12Device* device, const char* fbxPath, int nParticles) {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(fbxPath,
            aiProcess_Triangulate | aiProcess_LimitBoneWeights |
            aiProcess_JoinIdenticalVertices | aiProcess_SortByPType |
            aiProcess_CalcTangentSpace);
        if (!scene || scene->mNumMeshes == 0) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                "HipHopAnimation::createParticleResources: Assimp failed to load '%s': %s\n",
                fbxPath, importer.GetErrorString());
            OutputDebugStringA(msg);
            return false;
        }
        // Count only triangle-face meshes (aiProcess_SortByPType may produce non-triangle sub-meshes)
        unsigned int meshIdx = 0;
        for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi) {
            if (scene->mMeshes[mi]->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) { meshIdx = mi; break; }
        }

        XMStoreFloat4x4(&globalInvF,
            XMMatrixInverse(nullptr, aiToXM(scene->mRootNode->mTransformation)));

        nodes.reserve(256);
        buildNodeTree(scene->mRootNode, -1);
        globalTransforms.resize(nodes.size());

        const aiMesh* mesh = scene->mMeshes[meshIdx];
        nBones = (int)mesh->mNumBones;
        if (nBones > MAX_BONES) nBones = MAX_BONES;
        boneInfos.resize(nBones);

        std::vector<std::vector<std::pair<int,float>>> vertWeights(mesh->mNumVertices);
        for (int bi = 0; bi < nBones; ++bi) {
            const aiBone* ab = mesh->mBones[bi];
            XMStoreFloat4x4(&boneInfos[bi].offsetMatrix, aiToXM(ab->mOffsetMatrix));
            auto it = nodeNameToIdx.find(ab->mName.C_Str());
            boneInfos[bi].nodeIdx = (it != nodeNameToIdx.end()) ? it->second : 0;
            for (unsigned w = 0; w < ab->mNumWeights; ++w)
                vertWeights[ab->mWeights[w].mVertexId].push_back({bi, ab->mWeights[w].mWeight});
        }

        std::vector<SkinnedVertex> verts(mesh->mNumVertices);
        for (unsigned vi = 0; vi < mesh->mNumVertices; ++vi) {
            auto& sv = verts[vi];
            sv.pos[0] = mesh->mVertices[vi].x;
            sv.pos[1] = mesh->mVertices[vi].y;
            sv.pos[2] = mesh->mVertices[vi].z;
            auto& wl = vertWeights[vi];
            std::sort(wl.begin(), wl.end(), [](auto& x, auto& y){ return x.second > y.second; });
            if (wl.size() > 4) wl.resize(4);
            float wsum = 0.f; for (auto& p : wl) wsum += p.second;
            if (wsum < 1e-6f) wsum = 1.f;
            for (int k = 0; k < 4; ++k) {
                sv.boneIdx[k] = (k < (int)wl.size()) ? (uint8_t)wl[k].first : 0;
                sv.boneWt[k]  = (k < (int)wl.size()) ? wl[k].second / wsum  : 0.f;
                sv.norm[0]=sv.norm[1]=sv.norm[2]=sv.tangent[0]=sv.tangent[1]=sv.tangent[2]=sv.uv[0]=sv.uv[1]=0.f;
            }
        }

        std::vector<unsigned int> indices;
        indices.reserve(mesh->mNumFaces * 3);
        for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
            const aiFace& face = mesh->mFaces[fi];
            if (face.mNumIndices != 3) continue;
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }

        if (scene->mNumAnimations > 0) {
            const aiAnimation* anim = scene->mAnimations[0];
            animDuration    = anim->mDuration;
            animTicksPerSec = (anim->mTicksPerSecond > 1e-9) ? anim->mTicksPerSecond : 25.0;
            channels.resize(anim->mNumChannels);
            for (unsigned ci = 0; ci < anim->mNumChannels; ++ci) {
                const aiNodeAnim* na = anim->mChannels[ci];
                auto it2 = nodeNameToIdx.find(na->mNodeName.C_Str());
                if (it2 != nodeNameToIdx.end())
                    nodes[it2->second].animChannelIdx = (int)ci;
                Channel& ch = channels[ci];
                for (unsigned k = 0; k < na->mNumPositionKeys; ++k) {
                    ch.posTimes.push_back(na->mPositionKeys[k].mTime);
                    auto& v = na->mPositionKeys[k].mValue;
                    ch.posKeys.push_back({v.x, v.y, v.z});
                }
                for (unsigned k = 0; k < na->mNumRotationKeys; ++k) {
                    ch.rotTimes.push_back(na->mRotationKeys[k].mTime);
                    auto& q = na->mRotationKeys[k].mValue;
                    ch.rotKeys.push_back({q.x, q.y, q.z, q.w});
                }
                for (unsigned k = 0; k < na->mNumScalingKeys; ++k) {
                    ch.scaleTimes.push_back(na->mScalingKeys[k].mTime);
                    auto& v = na->mScalingKeys[k].mValue;
                    ch.scaleKeys.push_back({v.x, v.y, v.z});
                }
            }
        }

        sampleParticlesFromMesh(verts, indices, nParticles);

        // Create persistently-mapped upload heap buffers for per-frame GPU transfer
        UINT64 bufSize = (UINT64)nParticles * sizeof(XMFLOAT3);
        auto makeBuf = [&](com_ptr<ID3D12Resource>& buf, void** mapped) {
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(bufSize),
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(buf.GetAddressOf()));
            CD3DX12_RANGE rr(0, 0);
            buf->Map(0, &rr, mapped);
        };
        makeBuf(charPosUpload, reinterpret_cast<void**>(&charPosMapped));
        makeBuf(charVelUpload, reinterpret_cast<void**>(&charVelMapped));

        // Initialize buffers with T=0 pose so the first frame has valid data
        updateCharParticles(0.0f, 0.0f, XMFLOAT3(0, 0, 0));
        return true;
    }

    // Compute skinned world-space positions for this frame, write positions + velocities
    // to the persistently-mapped upload heap buffers. Call every frame before RecordComputeCommands.
    void updateCharParticles(float T, float dt, XMFLOAT3 modelTranslation, float modelScale = 1.0f) {
        if (charParticles.empty()) return;

        double ticks = fmod((double)T * animTicksPerSec, animDuration);
        if (ticks < 0.0) ticks += animDuration;
        computeGlobal(0, XMMatrixIdentity(), ticks);

        XMMATRIX globalInv = XMLoadFloat4x4(&globalInvF);
        XMVECTOR modelTransV = XMLoadFloat3(&modelTranslation);

        for (int p = 0; p < (int)charParticles.size(); p++) {
            const CharParticle& cp = charParticles[p];
            XMVECTOR bindPos = XMLoadFloat3(&cp.bindPos);
            XMVECTOR skinned = XMVectorZero();
            for (int k = 0; k < 4; k++) {
                if (cp.boneWt[k] < 1e-6f) continue;
                const BoneInfo& bi = boneInfos[cp.boneIdx[k]];
                XMMATRIX bm = XMLoadFloat4x4(&bi.offsetMatrix)
                            * globalTransforms[bi.nodeIdx]
                            * globalInv;
                skinned = XMVectorAdd(skinned, XMVectorScale(XMVector3Transform(bindPos, bm), cp.boneWt[k]));
            }
            skinned = XMVectorScale(skinned, modelScale);
            skinned = XMVectorAdd(skinned, modelTransV);

            XMFLOAT3 newPos;
            XMStoreFloat3(&newPos, skinned);

            XMFLOAT3 vel(0, 0, 0);
            if (dt > 1e-4f) {
                XMFLOAT3& prev = prevCharPos[p];
                vel = {(newPos.x - prev.x) / dt,
                       (newPos.y - prev.y) / dt,
                       (newPos.z - prev.z) / dt};
            }
            prevCharPos[p] = newPos;

            charPosMapped[p] = newPos;
            charVelMapped[p] = vel;
        }
    }

    ID3D12Resource* GetCharPosUpload() const { return charPosUpload.Get(); }
    ID3D12Resource* GetCharVelUpload() const { return charVelUpload.Get(); }
    int getNumCharParticles() const { return (int)charParticles.size(); }

    void releaseResources() {
        if (skinCbMapped) { skinCbBuffer->Unmap(0, nullptr); skinCbMapped = nullptr; }
        skinCbBuffer.Reset();
        geometry.reset();
        rootSigSkinned.Reset();
        rootSigSkinnedCollect.Reset();
        for (auto& p : pso) p.Reset();
    }
};
