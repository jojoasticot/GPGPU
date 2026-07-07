#include <cassert>
#include <chrono>
#include <cstdio>
#include <curand_kernel.h>
#include <thread>

#include "filter_impl.h"

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

__device__ int pixel_diff(rgb p1, rgb p2)
{
    int r = abs(p1.r - p2.r);
    int g = abs(p1.g - p2.g);
    int b = abs(p1.b - p2.b);
    return sqrtf(r * r + g * g + b * b);
}

__device__ int find_matching_reservoir(rgb p, reservoir* rs)
{
    int m_idx = -1;

    for (int i = 0; i < K; i++)
    {
        if (rs[i].w > 0)
        {
            if (abs(p.r - rs[i].rgb.r) < RGB_DIFF_THRESHOLD
                && abs(p.g - rs[i].rgb.g) < RGB_DIFF_THRESHOLD
                && abs(p.b - rs[i].rgb.b) < RGB_DIFF_THRESHOLD)
                return i;
        }
        else
            m_idx = i;
    }

    return m_idx;
}

__device__ void update_weight_and_samples(reservoir* rs, int m_idx, rgb p)
{
    rs[m_idx].w++;
    float w = rs[m_idx].w;

    float temp_r = ((w - 1) * rs[m_idx].rgb.r + p.r) / w;
    float temp_g = ((w - 1) * rs[m_idx].rgb.g + p.g) / w;
    float temp_b = ((w - 1) * rs[m_idx].rgb.b + p.b) / w;

    rs[m_idx].rgb = { static_cast<uint8_t>(temp_r),
                      static_cast<uint8_t>(temp_g),
                      static_cast<uint8_t>(temp_b) };
}

__global__ void compute_diff_kernel(std::byte* buffer, int width, int height,
                                    int stride, reservoir* rss, int* diff_map,
                                    curandState* states)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    rgb* lineptr = (rgb*)(buffer + y * stride);
    rgb p = lineptr[x];
    reservoir* rs = &rss[idx * K];

    int m_idx = find_matching_reservoir(p, rs);

    if (m_idx != -1 && rs[m_idx].w > 0)
        update_weight_and_samples(rs, m_idx, p);
    else if (m_idx != -1 && rs[m_idx].w == 0)
    {
        rs[m_idx].rgb = p;
        rs[m_idx].w = 1;
    }
    else
    {
        m_idx = 0;
        float total_weight = rs[0].w;

        for (int i = 1; i < K; i++)
        {
            if (rs[i].w < rs[m_idx].w)
                m_idx = i;

            total_weight += rs[i].w;
        }

        curandState local_state = states[idx];
        float rand_float = curand_uniform(&local_state);
        states[idx] = local_state;

        if (rand_float * total_weight >= rs[m_idx].w)
        {
            rs[m_idx].rgb = p;
            rs[m_idx].w = 1;
        }
    }

    for (int i = 0; i < K; i++)
        if (rs[i].w > MAX_WEIGHTS)
            rs[i].w = MAX_WEIGHTS;

    int max_idx = 0;

    for (int i = 1; i < K; i++)
        if (rs[max_idx].w <= rs[i].w)
            max_idx = i;

    diff_map[idx] = pixel_diff(p, rs[max_idx].rgb) / 3;
}

__global__ void init_curand_kernel(curandState* states, int width, int height,
                                   unsigned long long seed)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void erode_kernel(const int* in, int* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;
    int min_val = in[idx];

    if (y - 1 >= 0)
        min_val = min(min_val, in[idx - width]);

    if (x - 1 >= 0)
        min_val = min(min_val, in[idx - 1]);

    if (y + 1 < height)
        min_val = min(min_val, in[idx + width]);

    if (x + 1 < width)
        min_val = min(min_val, in[idx + 1]);

    out[idx] = min_val;
}

__global__ void dilate_kernel(const int* in, int* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;
    int max_val = in[idx];

    if (y - 1 >= 0)
        max_val = max(max_val, in[idx - width]);

    if (x - 1 >= 0)
        max_val = max(max_val, in[idx - 1]);

    if (y + 1 < height)
        max_val = max(max_val, in[idx + width]);

    if (x + 1 < width)
        max_val = max(max_val, in[idx + 1]);

    out[idx] = max_val;
}

__global__ void threshold_kernel(const int* opened, uint8_t* I_low,
                                 uint8_t* I_high, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    I_low[idx] = opened[idx] > T_LOW;
    I_high[idx] = opened[idx] > T_HIGH;
}

__global__ void hysteresis(const uint8_t* I_low, const uint8_t* I_high,
                           uint8_t* out, int width, int height, int* changed)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (out[idx] || !I_low[idx])
        return;

    if (I_high[idx])
    {
        out[idx] = 1;
        atomicExch(changed, 1);
        return;
    }

    if ((y > 0 && out[idx - width]) || (x > 0 && out[idx - 1])
        || (y + 1 < height && out[idx + width])
        || (x + 1 < width && out[idx + 1]))
    {
        out[idx] = 1;
        atomicExch(changed, 1);
    }
}

__global__ void paint_kernel(std::byte* buffer, int width, int height,
                             int stride, const uint8_t* mask)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    if (mask[y * width + x])
    {
        rgb* lineptr = (rgb*)(buffer + y * stride);
        lineptr[x].r = 255;
    }
}

extern "C"
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride,
                     int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        cudaError_t err;

        static std::byte* dBuffer = nullptr;
        static size_t dPitch = 0;
        static reservoir* rss = nullptr;
        static curandState* rand_states = nullptr;
        static int* diff_map = nullptr;
        static int* tmp = nullptr;
        static uint8_t* I_low = nullptr;
        static uint8_t* I_high = nullptr;
        static uint8_t* mask = nullptr;
        static int* changed = nullptr;

        dim3 blockSize(16, 16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                      (height + (blockSize.y - 1)) / blockSize.y);

        if (rss == nullptr)
        {
            err =
                cudaMallocPitch(&dBuffer, &dPitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);

            err = cudaMalloc(&rss, width * height * K * sizeof(reservoir));
            CHECK_CUDA_ERROR(err);
            err = cudaMemset(rss, 0, width * height * K * sizeof(reservoir));
            CHECK_CUDA_ERROR(err);

            err =
                cudaMalloc(&rand_states, width * height * sizeof(curandState));
            CHECK_CUDA_ERROR(err);
            init_curand_kernel<<<gridSize, blockSize>>>(rand_states, width,
                                                        height, 1234ULL);
            CHECK_CUDA_ERROR(cudaGetLastError());

            err = cudaMalloc(&diff_map, width * height * sizeof(int));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&tmp, width * height * sizeof(int));
            CHECK_CUDA_ERROR(err);

            err = cudaMalloc(&I_low, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&I_high, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&mask, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);

            err = cudaMalloc(&changed, sizeof(int));
            CHECK_CUDA_ERROR(err);
        }

        err = cudaMemcpy2D(dBuffer, dPitch, src_buffer, src_stride,
                           width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        compute_diff_kernel<<<gridSize, blockSize>>>(
            dBuffer, width, height, static_cast<int>(dPitch), rss, diff_map,
            rand_states);
        CHECK_CUDA_ERROR(cudaGetLastError());

        erode_kernel<<<gridSize, blockSize>>>(diff_map, tmp, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        dilate_kernel<<<gridSize, blockSize>>>(tmp, diff_map, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        threshold_kernel<<<gridSize, blockSize>>>(diff_map, I_low, I_high,
                                                  width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        err = cudaMemset(mask, 0, width * height * sizeof(uint8_t));
        CHECK_CUDA_ERROR(err);

        int host_changed = 1;
        while (host_changed)
        {
            err = cudaMemset(changed, 0, sizeof(int));
            CHECK_CUDA_ERROR(err);

            hysteresis<<<gridSize, blockSize>>>(I_low, I_high, mask, width,
                                                height, changed);
            CHECK_CUDA_ERROR(cudaGetLastError());

            err = cudaMemcpy(&host_changed, changed, sizeof(int),
                             cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERROR(err);
        }

        paint_kernel<<<gridSize, blockSize>>>(dBuffer, width, height,
                                              static_cast<int>(dPitch), mask);
        CHECK_CUDA_ERROR(cudaGetLastError());

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, dPitch,
                           width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);
    }
}
