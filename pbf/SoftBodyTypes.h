#pragma once

#include "Egg/Math/Float3.h"
#include "SharedConfig.hlsli"

using namespace Egg::Math;

// SoA field definitions for the BCC soft body dynamics nodes.
// Three fields: position (committed), velocity, and predicted position (used during constraint solving).
enum SbdField : unsigned int {
    SBD_POSITION = 0,
    SBD_VELOCITY,
    SBD_PREDICTED_POSITION,
    SBD_COUNT
};

static constexpr unsigned int sbdFieldStrides[SBD_COUNT] = {
    sizeof(Float3), // SBD_POSITION
    sizeof(Float3), // SBD_VELOCITY
    sizeof(Float3), // SBD_PREDICTED_POSITION
};

static constexpr const wchar_t* sbdFieldNames[SBD_COUNT] = {
    L"Sbd Position", L"Sbd Velocity", L"Sbd PredictedPosition"
};
