#include <curand_kernel.h>
#include <cuda_runtime.h>

extern "C" __global__
void mc_kernel(float *d_out,
               int   n_samples,
               float mu,
               float sigma,
               unsigned int seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    // ✅ Use o tipo correto
    curandState_t state;
    curand_init(seed_offset + idx, 0, 0, &state);

    float u1 = curand_uniform(&state);
    float u2 = curand_uniform(&state);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    d_out[idx] = mu + sigma * z0;
}
