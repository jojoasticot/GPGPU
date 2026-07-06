#include "filter_impl.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "logo.h"

template <typename T>
inline static T abs(T e)
{
    return e > 0 ? e : -e;
}

template <typename T>
inline static T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
inline static T min(T a, T b)
{
    return a < b ? a : b;
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

struct reservoir_set
{
    reservoir_set(int w, int h)
        : rss(w * h)
    {}

    reservoir* operator[](int i)
    {
        return rss[i];
    }

    int w, h;
    std::vector<reservoir[K]> rss;
};

extern "C"
{
    static int pixel_diff(rgb p1, rgb p2)
    {
        int diff_r = abs(static_cast<int>(p1.r) - static_cast<int>(p2.r));
        int diff_g = abs(static_cast<int>(p1.g) - static_cast<int>(p2.g));
        int diff_b = abs(static_cast<int>(p1.b) - static_cast<int>(p2.b));
        return std::sqrt(diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
    }

    static int find_matching_reservoir(rgb p, reservoir* rs)
    {
        int m_idx = -1;

        for (int i = 0; i < K; i++)
        {
            if (rs[i].w > 0)
            {
                if (abs(static_cast<int>(p.r) - static_cast<int>(rs[i].rgb.r))
                        < RGB_DIFF_THRESHOLD
                    && abs(static_cast<int>(p.g)
                           - static_cast<int>(rs[i].rgb.g))
                        < RGB_DIFF_THRESHOLD
                    && abs(static_cast<int>(p.b)
                           - static_cast<int>(rs[i].rgb.b))
                        < RGB_DIFF_THRESHOLD)
                    return i;
            }
            else
                m_idx = i;
        }

        return m_idx;
    }

    static inline void update_weight_and_samples(reservoir* rs, int m_idx,
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

    static std::vector<int> erosion_3(std::vector<int>& diff_map, int width,
                                      int height)
    {
        std::vector<int> erosed(width * height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int min_val = diff_map[y * width + x];

                if (y - 1 >= 0)
                    min_val = min(min_val, diff_map[(y - 1) * width + x]);

                if (x - 1 >= 0)
                    min_val = min(min_val, diff_map[y * width + (x - 1)]);

                if (y + 1 < height)
                    min_val = min(min_val, diff_map[(y + 1) * width + x]);

                if (x + 1 < width)
                    min_val = min(min_val, diff_map[y * width + (x + 1)]);

                erosed[y * width + x] = min_val;
            }
        }

        return erosed;
    }

    static std::vector<int> dilatation_3(std::vector<int>& diff_map, int width,
                                         int height)
    {
        std::vector<int> dilated(width * height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int max_val = diff_map[y * width + x];

                if (y - 1 >= 0)
                    max_val = max(max_val, diff_map[(y - 1) * width + x]);

                if (x - 1 >= 0)
                    max_val = max(max_val, diff_map[y * width + (x - 1)]);

                if (y + 1 < height)
                    max_val = max(max_val, diff_map[(y + 1) * width + x]);

                if (x + 1 < width)
                    max_val = max(max_val, diff_map[y * width + (x + 1)]);

                dilated[y * width + x] = max_val;
            }
        }

        return dilated;
    }

    static std::vector<int> opening_3(std::vector<int>& diff_map, int width,
                                      int height)
    {
        std::vector<int> erosed = erosion_3(diff_map, width, height);
        return dilatation_3(erosed, width, height);
    }

    static std::vector<bool> hysteresis(std::vector<bool>& input,
                                        std::vector<bool>& marker, int width,
                                        int height)
    {
        std::vector<bool> out(width * height);
        bool has_changed = true;

        while (has_changed)
        {
            has_changed = false;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int p = y * width + x;

                    if (out[p] || !input[p])
                        continue;

                    if (marker[p])
                    {
                        out[p] = true;
                        has_changed = true;
                        continue;
                    }

                    if (y > 0 && out[p - width])
                    {
                        has_changed = true;
                        out[p] = true;
                        continue;
                    }

                    if (x > 0 && out[p - 1])
                    {
                        has_changed = true;
                        out[p] = true;
                        continue;
                    }

                    if (y + 1 < height && out[p + width])
                    {
                        has_changed = true;
                        out[p] = true;
                        continue;
                    }

                    if (x + 1 < width && out[p + 1])
                    {
                        has_changed = true;
                        out[p] = true;
                        continue;
                    }
                }
            }
        }

        return out;
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        static reservoir_set rss(width, height);
        std::vector<int> diff_map(width * height);

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                rgb p = lineptr[x];
                reservoir* rs = rss[y * width + x];

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

                    float rand_float = (static_cast<double>(rand())
                                        / static_cast<double>(RAND_MAX));

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

                diff_map[y * width + x] =
                    pixel_diff(lineptr[x], rs[max_idx].rgb) / 3;
            }
        }

        std::vector<int> opened = opening_3(diff_map, width, height);
        std::vector<bool> I_low(width * height);
        std::vector<bool> I_high(width * height);

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);

            for (int x = 0; x < width; ++x)
            {
                I_low[y * width + x] = opened[y * width + x] > T_LOW;
                I_high[y * width + x] = opened[y * width + x] > T_HIGH;
            }
        }

        std::vector<bool> mask = hysteresis(I_low, I_high, width, height);

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x)
                if (mask[y * width + x])
                    // lineptr[x] = { 255, 255, 255 };
                    lineptr[x].r = 255;
        }
    }
} // "extern C"
