#pragma once
#include "Egg/Common.h"
#include <string>


class RawBuffer {
	com_ptr<ID3D12Resource>		buffer;
	com_ptr<ID3D12Resource>     uploadBuffer;
	com_ptr<ID3D12Resource>		readbackBuffer;
	uint bufferUintSize;
	std::wstring debugName;
public:
	RawBuffer(
		std::wstring debugName,
		uint bufferUintSize = 32 * 32 * 32 * 32) :debugName(debugName), bufferUintSize(bufferUintSize) {

	}

	void createResources(com_ptr<ID3D12Device> device,
		const D3D12_CPU_DESCRIPTOR_HANDLE& handle
	) {
		const D3D12_HEAP_PROPERTIES defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		const D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(4 * bufferUintSize,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 0);
		DX_API("commited resource")
			device->CreateCommittedResource(
				&defaultHeapProperties,
				//not needed for d3d11 interop, do not believe bogus warning: sharedWithD3D11?D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, 
				D3D12_HEAP_FLAG_NONE,
				&bufferDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				nullptr,
				IID_PPV_ARGS(buffer.ReleaseAndGetAddressOf()));
		buffer->SetName(debugName.c_str());

		CD3DX12_HEAP_PROPERTIES rbheapProps(D3D12_HEAP_TYPE_READBACK);

		D3D12_RESOURCE_ALLOCATION_INFO info = {};
		info.SizeInBytes = 4 * bufferUintSize;
		info.Alignment = 0;
		const D3D12_RESOURCE_DESC tempBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(info);
		DX_API("readback resource")
			device->CreateCommittedResource(
				&rbheapProps,
				D3D12_HEAP_FLAG_NONE,
				&tempBufferDesc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(readbackBuffer.ReleaseAndGetAddressOf()));
		readbackBuffer->SetName((debugName + L" [READBACK]").c_str());


		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);

		DX_API("upload resource")
			device->CreateCommittedResource(
				&heapProps,
				D3D12_HEAP_FLAG_NONE,
				&tempBufferDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(uploadBuffer.ReleaseAndGetAddressOf()));
		uploadBuffer->SetName((debugName + L" [UPLOAD]").c_str());

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.CounterOffsetInBytes = 0;
		uavDesc.Buffer.FirstElement = 0;
		uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
		uavDesc.Buffer.NumElements = bufferUintSize;
		uavDesc.Buffer.StructureByteStride = 0;
		// create uav
		device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uavDesc, handle);
	}

	void releaseResources() {
		buffer.Reset();
		uploadBuffer.Reset();
		readbackBuffer.Reset();
	}

	void copyBack(com_ptr<ID3D12GraphicsCommandList> commandList) {
		barrier(commandList, buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		commandList->CopyResource(readbackBuffer.Get(), buffer.Get());
		barrier(commandList, buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	}

	unsigned int* mapReadback() {
		D3D12_RANGE readbackBufferRange{ 0, 4 * 32 * 32 * 32 };
		unsigned int* pReadbackBufferData;
		readbackBuffer->Map
		(
			0,
			&readbackBufferRange,
			reinterpret_cast<void**>(&pReadbackBufferData)
		);
		return pReadbackBufferData;
	}
	void unmapReadback() {
		D3D12_RANGE emptyRange{ 0, 0 };
		readbackBuffer->Unmap
		(
			0,
			&emptyRange
		);
	}

	void fillLinear() {
		void* pData;
		CD3DX12_RANGE range(0, bufferUintSize);
		uploadBuffer->Map(0, &range, &pData);
		unsigned int* m_arrayDataBegin = reinterpret_cast<unsigned int*>(pData);
		unsigned int* m_arrayDataEnd = m_arrayDataBegin + bufferUintSize;

		for (auto ip = m_arrayDataBegin; ip < m_arrayDataEnd; ip++) {
			*ip = ip - m_arrayDataBegin;
		}
		uploadBuffer->Unmap(0, &range);
	}

	void fillFFFFFFFF() {
		void* pData;
		CD3DX12_RANGE range(0, bufferUintSize);
		uploadBuffer->Map(0, &range, &pData);
		unsigned int* m_arrayDataBegin = reinterpret_cast<unsigned int*>(pData);
		unsigned int* m_arrayDataEnd = m_arrayDataBegin + bufferUintSize;

		for (auto ip = m_arrayDataBegin; ip < m_arrayDataEnd; ip++) {
			*ip = 0xffffffff;
		}
		uploadBuffer->Unmap(0, &range);
	}

	void fillRandomMask(uint m) {
		void* pData;
		CD3DX12_RANGE range(0, bufferUintSize);
		uploadBuffer->Map(0, &range, &pData);
		unsigned int* m_arrayDataBegin = reinterpret_cast<unsigned int*>(pData);
		unsigned int* m_arrayDataEnd = m_arrayDataBegin + bufferUintSize;

		for (auto ip = m_arrayDataBegin; ip < m_arrayDataEnd; ip++) {
			*ip = rand() & m;
			*ip |= (rand() & m) << 8;
			*ip |= (rand() & m) << 16;
			*ip |= (rand() & m) << 24;
		}
		uploadBuffer->Unmap(0, &range);
	}

	void fillRandom() {
		void* pData;
		CD3DX12_RANGE range(0, bufferUintSize);
		uploadBuffer->Map(0, &range, &pData);
		unsigned int* m_arrayDataBegin = reinterpret_cast<unsigned int*>(pData);
		unsigned int* m_arrayDataEnd = m_arrayDataBegin + bufferUintSize;

		for (auto ip = m_arrayDataBegin; ip < m_arrayDataEnd; ip++) {
			*ip = rand() & 0xff;
			*ip |= (rand() & 0xff) << 8;
			*ip |= (rand() & 0xff) << 16;
			*ip |= (rand() & 0xff) << 24;
		}

		m_arrayDataBegin[0] = 0x1454abff;
		m_arrayDataBegin[1] = 0xa1667600;
		m_arrayDataBegin[2] = 0x23144bca;
		m_arrayDataBegin[3] = 0x004156fe;
		m_arrayDataBegin[4] = 0xf4541bff;
		m_arrayDataBegin[5] = 0xa5667100;
		m_arrayDataBegin[6] = 0x232441ca;
		m_arrayDataBegin[7] = 0x004156f1;
		m_arrayDataBegin[8] = 0x4454abfd;
		m_arrayDataBegin[9] = 0xa4667600;
		m_arrayDataBegin[10] = 0x23444bca;
		m_arrayDataBegin[11] = 0x004456fe;
		m_arrayDataBegin[12] = 0x34544bfb;
		m_arrayDataBegin[13] = 0xa4667400;
		m_arrayDataBegin[14] = 0x235444cb;
		m_arrayDataBegin[15] = 0x004656f4;
		m_arrayDataBegin[16] = 0xf4547bfc;
		m_arrayDataBegin[17] = 0xaf667800;
		m_arrayDataBegin[18] = 0x23f44b9a;
		m_arrayDataBegin[19] = 0x004f56fa;
		m_arrayDataBegin[20] = 0xf454fbff;
		m_arrayDataBegin[21] = 0xa5667f09;
		m_arrayDataBegin[22] = 0x3243fca;
		m_arrayDataBegin[23] = 0xc04156ff;
		m_arrayDataBegin[24] = 0xfc54abf8;
		m_arrayDataBegin[25] = 0xa5c67606;
		m_arrayDataBegin[26] = 0x232c4b57;
		m_arrayDataBegin[27] = 0x0041c4fe;
		m_arrayDataBegin[28] = 0xf4543cf5;
		m_arrayDataBegin[29] = 0xa56976c0;
		m_arrayDataBegin[30] = 0x23844bc3;
		m_arrayDataBegin[31] = 0x074156f2;

		uploadBuffer->Unmap(0, &range);
	}

	static void barrier(com_ptr<ID3D12GraphicsCommandList> pCmdList, com_ptr<ID3D12Resource> pResource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after, D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE)
	{
		D3D12_RESOURCE_BARRIER barrierDesc = {};

		barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrierDesc.Flags = flags;
		barrierDesc.Transition.pResource = pResource.Get();
		barrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrierDesc.Transition.StateBefore = before;
		barrierDesc.Transition.StateAfter = after;

		pCmdList->ResourceBarrier(1, &barrierDesc);
	}

	void upload(com_ptr<ID3D12GraphicsCommandList> commandList) {
		barrier(commandList, buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);

		commandList->CopyResource(buffer.Get(), uploadBuffer.Get());

		barrier(commandList, buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}

	D3D12_RESOURCE_BARRIER uavBarrier() const {
		D3D12_RESOURCE_BARRIER b;
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		b.UAV.pResource = buffer.Get();
		return b;
	}

};