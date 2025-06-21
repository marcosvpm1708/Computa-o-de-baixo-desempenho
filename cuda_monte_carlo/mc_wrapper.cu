#include <cuda_runtime.h>
#include <stdio.h>
#include "mc_kernel.h"

extern "C" void launch_mc(int n_samples,
                          float mu,
                          float sigma,
                          unsigned int seed_offset,
                          float *h_out) {
    printf("[wrapper-test] launch_mc called\n");
    float *d_out;
    size_t bytes = n_samples * sizeof(float);
    cudaMalloc(&d_out, bytes);

    int blockSize = 256;
    int gridSize  = (n_samples + blockSize - 1) / blockSize;

    mc_kernel<<<gridSize, blockSize>>>(d_out, n_samples, mu, sigma, seed_offset);
    cudaDeviceSynchronize();

    float first;
    cudaMemcpy(&first, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[wrapper-test] after kernel, sample[0]=%.6f\n", first);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("[wrapper-test] copied host_buf[0]=%.6f\n", h_out[0]);

    cudaFree(d_out);
}
