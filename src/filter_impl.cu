#include <cassert>
#include <chrono>
#include <cmath>
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

#define ABS(a) ((a) > 0 ? (a) : (-a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
    __device__ int pixel_diff(rgb p1, rgb p2)
    {
        int diff_r = ABS(static_cast<int>(p1.r) - static_cast<int>(p2.r));
        int diff_g = ABS(static_cast<int>(p1.g) - static_cast<int>(p2.g));
        int diff_b = ABS(static_cast<int>(p1.b) - static_cast<int>(p2.b));
        return std::sqrt(diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
    }

    __device__ int find_matching_reservoir(rgb p, reservoir* rs)
    {
        int m_idx = -1;

        for (int i = 0; i < K; i++)
        {
            if (rs[i].w > 0)
            {
                if (ABS(static_cast<int>(p.r) - static_cast<int>(rs[i].rgb.r))
                        < RGB_DIFF_THRESHOLD
                    && ABS(static_cast<int>(p.g)
                           - static_cast<int>(rs[i].rgb.g))
                        < RGB_DIFF_THRESHOLD
                    && ABS(static_cast<int>(p.b)
                           - static_cast<int>(rs[i].rgb.b))
                        < RGB_DIFF_THRESHOLD)
                    return i;
            }
            else
                m_idx = i;
        }

        return m_idx;
    }

    __device__ inline void update_weight_and_samples(reservoir* rs, int m_idx,
                                                     rgb p)
    {
        rs[m_idx].w++;
        float w = rs[m_idx].w;

        float temp_r = ((w - 1) * static_cast<float>(rs[m_idx].rgb.r)
                        + static_cast<float>(p.r))
            / w;
        float temp_g = ((w - 1) * static_cast<float>(rs[m_idx].rgb.g)
                        + static_cast<float>(p.g))
            / w;
        float temp_b = ((w - 1) * static_cast<float>(rs[m_idx].rgb.b)
                        + static_cast<float>(p.b))
            / w;

        rs[m_idx].rgb = { static_cast<uint8_t>(temp_r),
                          static_cast<uint8_t>(temp_g),
                          static_cast<uint8_t>(temp_b) };
    }

    __global__ void compute_diff(std::byte* buffer, int width, int height,
                                 int stride, reservoir* rss, int* diff_map)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= width || y >= height)
            return;

        rgb* lineptr = (rgb*)(buffer + y * stride);

        rgb p = lineptr[x];
        reservoir* rs = &rss[(y * width + x) * K];

        int m_idx = find_matching_reservoir(p, rs);

        if (m_idx != -1 && rs[m_idx].w > 0) // matching
            update_weight_and_samples(rs, m_idx, p);
        else if (m_idx != -1 && rs[m_idx].w == 0) // matching but empty
        {
            rs[m_idx].rgb = p;
            rs[m_idx].w = 1;
        }
        else // no match
        {
            m_idx = 0;
            float total_weight = rs[0].w;

            for (int i = 1; i < K; i++)
            {
                if (rs[i].w < rs[m_idx].w)
                    m_idx = i;

                total_weight += rs[i].w;
            }

            // TODO good random
            float rand_float =
                (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

            // 1 - rs[m_idx].w / total_weight chance to reset reservoir
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

        // apply max_id color to pixel, because this is the background

        diff_map[y * width + x] = pixel_diff(lineptr[x], rs[max_idx].rgb) / 3;
    }

    __global__ void(float* rand_states, int width, int height)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= width || y >= height)
            return;

        rand_states[y * width + x] = // TODO
    }

    __host__ void filter_impl(uint8_t* src_buffer, int width, int height,
                              int src_stride, int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        cudaError_t err;

        static reservoir* reservoir_set = nullptr;
        static float* rand_states = nullptr;

        dim3 blockSize(16, 16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                      (height + (blockSize.y - 1)) / blockSize.y);

        // init reservoir and rand_states for first frame
        if (reservoir_set == nullptr)
        {
            err = cudaMalloc(&reservoir_set,
                             width * height * K * sizeof(reservoir));
            CHECK_CUDA_ERROR(err);
            err = cudaMemset(reservoir_set, 0, width * height);
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&rand_states, width * height * sizeof(float));
            CHECK_CUDA_ERROR(err);
            curand_init<<<gridsize, blocksize>>>(rand_states, width, height);
        }

        // device copy of image
        std::byte* dBuffer;

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride,
                           width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        int* diff_map;

        err = cudaMalloc(&diff_map, width * height * sizeof(int));
        CHECK_CUDA_ERROR(err);

        compute_diff<<<gridSize, blockSize>>>(dBuffer, width, height, pitch,
                                              reservoir_set, diff_map);

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
