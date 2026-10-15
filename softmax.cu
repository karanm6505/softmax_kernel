#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)



__global__ void softmax_v1(const float* __restrict__ x, float* __restrict__ y,
                           int rows, int cols) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    float m = -INFINITY;
    for (int j = 0; j < cols; ++j) {
        m = fmaxf(xr[j], m);
    }

    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        s += expf(xr[j] - m);
    }

    float onebys = 1.0f / s;
    for (int j = 0; j < cols; ++j) {
        yr[j] = expf(xr[j] - m) * onebys;
    }
}
__global__ void softmax_v2(const float* __restrict__ x, float* __restrict__ y,
                           int rows, int cols) {
    __shared__ float buf[256];

    int row = blockIdx.x;       // one block per row
    int tid = threadIdx.x;      // 0..255

    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    // Pass 1: each thread's partial max, then tree-reduce to the row max
    float m = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        m = fmaxf(m, xr[j]);
    }
    buf[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[tid] = fmaxf(buf[tid], buf[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = buf[0];
    __syncthreads();            // all threads have read buf[0] before buf is reused

    // Pass 2: partial sums of exp(x - max), same reduction with +
    float s = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        s += expf(xr[j] - row_max);
    }
    buf[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    float onebys = 1.0f / buf[0];

    // Pass 3: write the output, same coalesced striding
    for (int j = tid; j < cols; j += blockDim.x) {
        yr[j] = expf(xr[j] - row_max) * onebys;
    }
}

__global__ void copy_rows(const float* __restrict__ x, float* __restrict__ y, int rows, int cols){

    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;

    for(int j = threadIdx.x; j < cols; j += blockDim.x){
        yr[j] = xr[j];
    }
}


__device__ float warp_max(float v){
    for(int offset = 16; offset > 0; offset /= 2)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

__device__ float warp_sum(float v){
    for(int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__global__ void softmax_v3(const float* __restrict__ x, float* __restrict__ y,
                           int rows, int cols) {
    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * cols;
    float* yr = y + (size_t)row * cols;

    float m = -INFINITY;
    for (int j = lane; j < cols; j += 32) m = fmaxf(m, xr[j]);
    m = warp_max(m);

    float s = 0.0f;
    for (int j = lane; j < cols; j += 32) s += expf(xr[j] - m);
    s = warp_sum(s);

    float onebys = 1.0f / s;
    for (int j = lane; j < cols; j += 32) yr[j] = expf(xr[j] - m) * onebys;
}

__device__ void combine(float &m, float &s, float m2, float s2){
    float m_new = fmaxf(m, m2);
    if(m_new == -INFINITY) return;
    s = s * expf(m - m_new) + s2 * expf(m2 - m_new);
    m = m_new;
}

__global__ void softmax_v4(const float* __restrict__ x, float* __restrict__ y, int rows, int cols){
    __shared__ float buf_m[256];
    __shared__ float buf_s[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* xr = x + (size_t) row * cols;
    float* yr = y + (size_t) row * cols;

    float m = -INFINITY, s = 0.0f;
        for (int j = tid; j < cols; j += blockDim.x) {
            float v = xr[j];
            float m_new = fmaxf(m, v);
            s = s * expf(m - m_new) + expf(v - m_new);
            m = m_new;
        }

        // Combine 256 (m, s) pairs: same tree as v2
        buf_m[tid] = m;
        buf_s[tid] = s;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                float mm = buf_m[tid], ss = buf_s[tid];
                combine(mm, ss, buf_m[tid + stride], buf_s[tid + stride]);
                buf_m[tid] = mm;
                buf_s[tid] = ss;
            }
            __syncthreads();
        }
        float row_max = buf_m[0];
        float onebys = 1.0f / buf_s[0];

        // Read #2: write the output
        for (int j = tid; j < cols; j += blockDim.x) {
            yr[j] = expf(xr[j] - row_max) * onebys;
        }
}

__global__ void softmax_v5(const float* __restrict__ x, float* __restrict__ y, int rows, int cols){
    __shared__ float buf_m[1024];
    __shared__ float buf_s[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* xr = x + (size_t) row * cols;
    float* yr = y + (size_t) row * cols;

    float m = -INFINITY, s = 0.0f;
        for (int j = tid; j < cols; j += blockDim.x) {
            float v = xr[j];
            float m_new = fmaxf(m, v);
            s = s * expf(m - m_new) + expf(v - m_new);
            m = m_new;
        }

        // Combine 1024 (m, s) pairs: same tree as v2
        buf_m[tid] = m;
        buf_s[tid] = s;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                float mm = buf_m[tid], ss = buf_s[tid];
                combine(mm, ss, buf_m[tid + stride], buf_s[tid + stride]);
                buf_m[tid] = mm;
                buf_s[tid] = ss;
            }
            __syncthreads();
        }
        float row_max = buf_m[0];
        float onebys = 1.0f / buf_s[0];

        // Read #2: write the output
        for (int j = tid; j < cols; j += blockDim.x) {
            yr[j] = expf(xr[j] - row_max) * onebys;
        }
}
void launch_copy(const float* x, float* y, int rows, int cols){
    copy_rows<<<rows, 256>>> (x, y, rows, cols);
}
void launch_v2(const float* x, float* y, int rows, int cols) {
    softmax_v2<<<rows, 256>>>(x, y, rows, cols);   // one block per row
}

void launch_v1(const float* x, float* y, int rows, int cols) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_v1<<<blocks, threads>>>(x, y, rows, cols);
}

void launch_v3(const float* x, float* y, int rows, int cols){
    int threads = 256;
    int rows_per_block = threads/32; // 8 warps -> 8 rows per blocks
    int blocks = (rows + rows_per_block - 1) / rows_per_block;
    softmax_v3<<<blocks, threads>>> (x, y, rows, cols);
}

void launch_v4(const float* x, float* y, int rows, int cols){
    softmax_v4<<<rows, 256>>> (x, y, rows, cols);
}
void launch_v5(const float* x, float* y, int rows, int cols){
    softmax_v5<<<rows, 1024>>> (x, y, rows, cols);
}

void launch_best(const float* x, float* y, int rows, int cols) {
    if (cols <= 1024)      launch_v3(x, y, rows, cols);   // warp per row
    else if (cols <= 4096) launch_v4(x, y, rows, cols);   // online, 256 threads
    else                   launch_v5(x, y, rows, cols);   // online, 1024 threads
}


struct Kernel {
    const char* name;
    void (*launch)(const float*, float*, int, int);
};

static const Kernel kKernels[] = {
    {"v1", launch_v1},
    {"v2", launch_v2},
    {"v3", launch_v3},
    {"v4", launch_v4},
    {"v5", launch_v5},
    {"best", launch_best},
    {"copy", launch_copy},
};


static void softmax_cpu(const std::vector<float>& x, std::vector<float>& y,
                        int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + (size_t)r * cols;
        float* yr = y.data() + (size_t)r * cols;
        double m = -INFINITY;
        for (int j = 0; j < cols; ++j) m = std::max(m, (double)xr[j]);
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += std::exp((double)xr[j] - m);
        for (int j = 0; j < cols; ++j) yr[j] = (float)(std::exp((double)xr[j] - m) / s);
    }
}

int main(int argc, char** argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 4096;
    int cols = argc > 2 ? atoi(argv[2]) : 1024;
    int iters = argc > 3 ? atoi(argv[3]) : 50;
    const char* only = argc > 4 ? argv[4] : nullptr;
    if (rows <= 0 || cols <= 0 || iters <= 0) {
        fprintf(stderr, "usage: %s <rows> <cols> [iters] [kernel]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# device: %s | rows=%d cols=%d iters=%d\n",
            prop.name, rows, cols, iters);

    const size_t n = (size_t)rows * cols;
    const size_t bytes = n * sizeof(float);

    // Inputs in [-8, 8]: wide enough that skipping the max-subtraction would
    // lose precision, small enough that nothing overflows.
    std::vector<float> hx(n), hy(n), href(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (auto& v : hx) v = dist(rng);
    softmax_cpu(hx, href, rows, cols);

    float *dx = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const double ideal_bytes = 2.0 * (double)bytes;  // read x once + write y once

    for (const Kernel& k : kKernels) {
        if (only && strcmp(only, k.name) != 0) continue;

        // --- correctness ---
        CUDA_CHECK(cudaMemset(dy, 0, bytes));
        k.launch(dx, dy, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs((double)hy[i] - (double)href[i]);
            double tol = 1e-6 + 1e-4 * std::fabs((double)href[i]);
            if (!(d <= tol)) ok = false;  // written this way so NaN fails
            if (d > max_err || std::isnan(d)) max_err = d;
        }

        for (int w = 0; w < 3; ++w) k.launch(dx, dy, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(iters);
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaEventRecord(start));
            k.launch(dx, dy, rows, cols);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[it], start, stop));
        }
        CUDA_CHECK(cudaGetLastError());
        std::sort(times.begin(), times.end());
        float ms = times[iters / 2];
        double gbps = ideal_bytes / (ms * 1e-3) / 1e9;

        printf("%s,%d,%d,%.4f,%.1f,%.2e,%s\n", k.name, rows, cols, ms, gbps,
               max_err, ok ? "PASS" : "FAIL");
        fflush(stdout);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
