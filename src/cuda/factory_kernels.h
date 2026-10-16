#ifndef FACTORY_KERNELS_H
#define FACTORY_KERNELS_H
#include "../data_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    cudaError_t launch_pack_strided(
        const void *source, void *destination, size_t elements, size_t element_size,
        const int *shape, const size_t *strides, int ndims);

    cudaError_t launch_tile_strided(
        const void *source, void *destination, size_t input_elements,
        size_t output_elements, size_t element_size, const int *shape,
        const size_t *strides, int ndims);

    void launch_bernoulli_kernel(
        float *values,
        bool *base,
        size_t total_elements,
        float p);
        
    void launch_assign_scalar_val_kernel(
        void *base,
        dtype_t dtype,
        scalar_value_t value,
        size_t total_elements);

    cudaError_t launch_validate_finite_kernel(
        const void *base,
        dtype_t dtype,
        size_t total_elements,
        int *invalid);

    cudaError_t launch_scale_uniform_range_kernel(
        void *base,
        dtype_t dtype,
        size_t total_elements,
        double minimum,
        double maximum);

    void launch_scale_range_kernel(
        float *values,
        void *base,
        dtype_t dtype,
        scalar_value_t min,
        scalar_value_t max,
        size_t total_elements);

    cudaError_t launch_arange_kernel(
        void *base,
        dtype_t dtype,
        size_t total_elements,
        double start,
        double step);

    cudaError_t launch_linspace_kernel(
        void *base,
        dtype_t dtype,
        size_t total_elements,
        double start,
        double stop,
        bool endpoint);

    cudaError_t launch_logspace_kernel(
        void *base,
        dtype_t dtype,
        size_t total_elements,
        double start,
        double stop,
        double base_value,
        bool endpoint);

    cudaError_t launch_eye_kernel(
        void *base,
        dtype_t dtype,
        size_t total_elements,
        int rows,
        int cols,
        int64_t offset);

#ifdef __cplusplus
}
#endif

#endif
