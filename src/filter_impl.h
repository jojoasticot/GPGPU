#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void filter_impl(uint8_t* buffer, int width, int height, int plane_stride,
                     int pixel_stride);

#ifdef __cplusplus
}
#endif
