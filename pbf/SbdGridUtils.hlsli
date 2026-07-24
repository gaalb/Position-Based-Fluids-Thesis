#ifndef SBD_GRID_UTILS_HLSLI
#define SBD_GRID_UTILS_HLSLI

#include "SharedConfig.hlsli"

// SBD spatial grid: same GRID_DIM^3 cell count as the PBF grid, but cell width = SBD_H.
// With SBD_H = 2.0 and GRID_DIM = 128, the grid covers ±128 m per axis — well beyond
// the simulation box. The wider cells mean a ±1-cell neighbor search (27 cells) captures
// everything within SBD_H, matching the coarser BCC lattice spacing.
// Row-major indexing (no Morton codes) — the SBD grid is sparse enough that cache
// locality from Morton ordering gives negligible benefit.

static const float SBD_GRID_HALF_EXTENT = GRID_DIM * SBD_H * 0.5f;
static const float3 SBD_GRID_MIN = float3(-SBD_GRID_HALF_EXTENT, -SBD_GRID_HALF_EXTENT, -SBD_GRID_HALF_EXTENT);

int3 sbdPosToCell(float3 pos)
{
    return clamp(int3((pos - SBD_GRID_MIN) / SBD_H),
                 int3(0, 0, 0),
                 int3(GRID_DIM - 1, GRID_DIM - 1, GRID_DIM - 1));
}

uint sbdCellIndex(int3 cell)
{
    return (uint)cell.x + (uint)cell.y * GRID_DIM + (uint)cell.z * GRID_DIM * GRID_DIM;
}

struct SbdNeighborCells {
    uint indices[27]; // 3x3x3 — one cell per SBD_H in each direction is sufficient
    uint count;
};

SbdNeighborCells SbdNeighborCellIndices(float3 pos)
{
    SbdNeighborCells result;
    result.count = 0;
    int3 myCell = sbdPosToCell(pos);
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        int3 nc = myCell + int3(dx, dy, dz);
        if (nc.x < 0 || nc.x >= GRID_DIM ||
            nc.y < 0 || nc.y >= GRID_DIM ||
            nc.z < 0 || nc.z >= GRID_DIM)
            continue;
        result.indices[result.count++] = sbdCellIndex(nc);
    }
    return result;
}

#endif // SBD_GRID_UTILS_HLSLI
