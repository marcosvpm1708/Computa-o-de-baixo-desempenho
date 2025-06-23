// mc_kernel.h
#ifndef MC_KERNEL_H
#define MC_KERNEL_H

extern "C" __global__
void mc_kernel(float          *d_out,
               int             n_samples,
               float           mu,
               float           sigma,
               unsigned int    seed_offset);

#endif
