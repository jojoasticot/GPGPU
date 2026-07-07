#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "filter_impl.h"
#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

constexpr int RGB_DIFF_THRESHOLD = 40;
constexpr int K = 7;
constexpr int MAX_WEIGHTS = 150;
constexpr int T_LOW = 20;
constexpr int T_HIGH = 50;

struct rgb
{
    uint8_t r, g, b;
};

struct reservoir
{
    struct rgb rgb;
    float w;
};

__constant__ uint8_t* logo;

__device__ int pixel_diff(rgb p1, rgb p2)
{
  int r = abs(p1.r - p2.r);
  int g = abs(p1.g - p2.g);
  int b = abs(p1.b - p2.b);
  return sqrtf(r*r + g*g + b*b);
}


__device__ int find_matching_reservoir(rgb p, reservoir* rs)
{
    int m_idx = -1;

    for (int i = 0; i < K; i++)
    {
        if (rs[i].w > 0)
        {
          if(abs(p.r - rs[i].rgb.r) < RGB_DIFF_THRESHOLD && 
            abs(p.g - rs[i].rgb.g) < RGB_DIFF_THRESHOLD && 
            abs(p.b - rs[i].rgb.b) < RGB_DIFF_THRESHOLD)
            return i;
        }
        else
            m_idx = i;
    }

    return m_idx;
}

__device__ void update_weight_and_samples(reservoir* rs, int m_idx,
                                                 rgb p)
{
    rs[m_idx].w++;
    float w = rs[m_idx].w;

    float temp_r = ((w - 1) * rs[m_idx].rgb.r
                    + p.r)
        / w;
    float temp_g = ((w - 1) * rs[m_idx].rgb.g
                    + p.g)
        / w;
    float temp_b = ((w - 1) * rs[m_idx].rgb.b
                    + p.b)
        / w;

    rs[m_idx].rgb = { temp_r,
                      temp_g,
                      temp_b };
}

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void remove_red_channel_inp(std::byte* buffer, int width, int height,
                                       int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* lineptr = (rgb*)(buffer + y * stride);
    if (y < logo_height && x < logo_width)
    {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
    }
    else
    {
        lineptr[x].r = 0;
    }
}

namespace
{
    void load_logo()
    {
        static auto buffer =
            std::unique_ptr<std::byte, decltype(&cudaFree)>{ nullptr,
                                                             &cudaFree };

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte* ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height,
                             cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }
    }
} // namespace

extern "C"
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride,
                     int pixel_stride)
    {
        load_logo();

        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        size_t pitch;

        cudaError_t err;

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride,
                           width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                      (height + (blockSize.y - 1)) / blockSize.y);

        remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height,
                                                        pitch);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch,
                           width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}
