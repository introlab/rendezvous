#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <cmath>
#include "utils/math/MathConstants.h"

namespace
{
    const int TABLE_SIZE = 65536;
    const int QUARTER_TABLE_SIZE = 65536 / 4;
    const int LAST_TABLE_INDEX = TABLE_SIZE - 1;
    const float PI_2 = 2.f * math::PI;
}

float precomputed_sin[TABLE_SIZE];

struct table_filler 
{
    table_filler() 
    {
        for (int i=0; i < TABLE_SIZE; i++) 
        {
            precomputed_sin[i] = std::sin(i*PI_2/TABLE_SIZE);
        }
    }
} table_filler_instance;

namespace math
{
inline float tsin(float x) { return precomputed_sin[int((x*65535.f)/PI_2) & LAST_TABLE_INDEX]; }
inline float tcos(float x) { return precomputed_sin[(int((x*65535.f)/PI_2) + QUARTER_TABLE_SIZE) & LAST_TABLE_INDEX]; }
}

#endif //!PERFORMANCE_H