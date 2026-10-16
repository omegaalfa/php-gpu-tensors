#include <cuda_runtime.h>
#include "factory_kernels.cuh"
#include "launch_config.cuh"
#include "dispatcher.h"
#include "../data_types.h"

struct transfer_layout
{
    int shape[10];
    size_t strides[10];
    int ndims;
};

template <typename T>
__global__ void pack_strided_kernel(const T *source, T *destination, size_t elements, transfer_layout layout)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < elements; i += (size_t)blockDim.x * gridDim.x)
    {
        size_t remaining = i, offset = 0;
        for (int d = layout.ndims - 1; d >= 0; d--)
        {
            offset += (remaining % layout.shape[d]) * layout.strides[d];
            remaining /= layout.shape[d];
        }
        destination[i] = source[offset];
    }
}

extern "C" cudaError_t launch_pack_strided(
    const void *source, void *destination, size_t elements, size_t element_size,
    const int *shape, const size_t *strides, int ndims)
{
    if (!elements) return cudaSuccess;
    transfer_layout layout;
    if (ndims < 1 || ndims > (int)(sizeof(layout.shape) / sizeof(layout.shape[0])))
        return cudaErrorInvalidValue;
    layout.ndims = ndims;
    for (int d = 0; d < ndims; d++)
    {
        layout.shape[d] = shape[d];
        layout.strides[d] = strides[d];
    }
    dim3 grid = cuda_grid_1d(elements);
    switch (element_size)
    {
        case 1: pack_strided_kernel<<<grid, 256>>>((const uint8_t *)source, (uint8_t *)destination, elements, layout); break;
        case 2: pack_strided_kernel<<<grid, 256>>>((const uint16_t *)source, (uint16_t *)destination, elements, layout); break;
        case 4: pack_strided_kernel<<<grid, 256>>>((const uint32_t *)source, (uint32_t *)destination, elements, layout); break;
        case 8: pack_strided_kernel<<<grid, 256>>>((const uint64_t *)source, (uint64_t *)destination, elements, layout); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

template <typename T>
__global__ void tile_strided_kernel(const T *source, T *destination,
                                    size_t input_elements, size_t output_elements,
                                    transfer_layout layout)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < output_elements;
         i += (size_t)blockDim.x * gridDim.x)
    {
        size_t remaining = i % input_elements;
        size_t offset = 0;
        for (int d = layout.ndims - 1; d >= 0; d--)
        {
            offset += (remaining % layout.shape[d]) * layout.strides[d];
            remaining /= layout.shape[d];
        }
        destination[i] = source[offset];
    }
}

extern "C" cudaError_t launch_tile_strided(
    const void *source, void *destination, size_t input_elements,
    size_t output_elements, size_t element_size, const int *shape,
    const size_t *strides, int ndims)
{
    if (!output_elements) return cudaSuccess;
    if (!source || !destination || !input_elements || !shape || !strides ||
        ndims < 1 || ndims > 10)
        return cudaErrorInvalidValue;

    transfer_layout layout;
    layout.ndims = ndims;
    for (int d = 0; d < ndims; d++)
    {
        if (shape[d] <= 0) return cudaErrorInvalidValue;
        layout.shape[d] = shape[d];
        layout.strides[d] = strides[d];
    }

    dim3 grid = cuda_grid_1d(output_elements);
    switch (element_size)
    {
        case 1: tile_strided_kernel<<<grid, 256>>>((const uint8_t *)source, (uint8_t *)destination, input_elements, output_elements, layout); break;
        case 2: tile_strided_kernel<<<grid, 256>>>((const uint16_t *)source, (uint16_t *)destination, input_elements, output_elements, layout); break;
        case 4: tile_strided_kernel<<<grid, 256>>>((const uint32_t *)source, (uint32_t *)destination, input_elements, output_elements, layout); break;
        case 8: tile_strided_kernel<<<grid, 256>>>((const uint64_t *)source, (uint64_t *)destination, input_elements, output_elements, layout); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

__global__ void bernoulli_kernel(
    const float *values,
    bool *output_data,
    size_t size,
    float p)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_data[idx] = values[idx] <= p;
    }
}

extern "C" void launch_bernoulli_kernel(
    float *values,
    bool *base,
    size_t total_elements,
    float p)
{
    if (total_elements == 0)
        return;

    bernoulli_kernel<<<cuda_grid_1d(total_elements), 256>>>(values, base, total_elements, p);
}

template <typename T>
__global__ void validate_finite_kernel(const T *values, size_t size, int *invalid)
{
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        if (!isfinite(values[idx]))
            atomicExch(invalid, 1);
    }
}

extern "C" cudaError_t launch_validate_finite_kernel(
    const void *base,
    dtype_t dtype,
    size_t total_elements,
    int *invalid)
{
    if (total_elements == 0)
        return cudaSuccess;
    if (!base || !invalid)
        return cudaErrorInvalidValue;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            validate_finite_kernel<<<grid, 256>>>((const float *)base, total_elements, invalid);
            break;
        case DTYPE_FLOAT64:
            validate_finite_kernel<<<grid, 256>>>((const double *)base, total_elements, invalid);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

template <typename T>
__global__ void scale_uniform_range_kernel(
    T *values,
    size_t size,
    double minimum,
    double maximum)
{
    T lower = (T)minimum;
    T upper = (T)maximum;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        double unit = (double)values[idx];
        /* cuRAND returns (0, 1]. Rotating its endpoint onto zero gives the
         * uniform finite grid [0, 1) without assigning extra mass to the
         * largest representable value below maximum. */
        if (unit == 1.0)
            unit = 0.0;

        double scaled = minimum * (1.0 - unit) + maximum * unit;
        T result = (T)scaled;

        if (!(result >= lower))
            result = lower;
        else if (result >= upper)
            /* Only rounding in the selected dtype can reach maximum here. */
            result = nextafter(upper, lower);
        values[idx] = result;
    }
}

extern "C" cudaError_t launch_scale_uniform_range_kernel(
    void *base,
    dtype_t dtype,
    size_t total_elements,
    double minimum,
    double maximum)
{
    if (total_elements == 0)
        return cudaSuccess;
    if (!base || !(minimum < maximum))
        return cudaErrorInvalidValue;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            scale_uniform_range_kernel<<<grid, 256>>>((float *)base, total_elements, minimum, maximum);
            break;
        case DTYPE_FLOAT64:
            scale_uniform_range_kernel<<<grid, 256>>>((double *)base, total_elements, minimum, maximum);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

extern "C" void launch_assign_scalar_val_kernel(
    void *base,
    dtype_t dtype,
    scalar_value_t value,
    size_t total_elements)
{
    DISPATCH_DTYPE(dtype, {
        scalar_t scalar;

        switch (dtype)
        {
        case DTYPE_FLOAT32:
            scalar = (scalar_t)value.v.f32;
            break;
        case DTYPE_FLOAT64:
            scalar = (scalar_t)value.v.f64;
            break;
        case DTYPE_INT32:
            scalar = (scalar_t)value.v.i32;
            break;
        case DTYPE_INT64:
            scalar = (scalar_t)value.v.i64;
            break;
        case DTYPE_INT8:
            scalar = (scalar_t)value.v.i8;
            break;
        case DTYPE_BOOL:
            scalar = (scalar_t)value.v.b;
            break;
        default:
            scalar = (scalar_t)0;
            break;
        }

        launch_fill_kernel_with_scalar<scalar_t>(
            (scalar_t *)base,
            scalar,
            total_elements);
    });
}

extern "C" void launch_scale_range_kernel(
    float *values,
    void *base,
    dtype_t dtype,
    scalar_value_t min,
    scalar_value_t max,
    size_t total_elements)
{
    if (total_elements == 0)
        return;

    DISPATCH_DTYPE(dtype, {
        scalar_t s_min, s_max;

        switch (dtype)
        {
        case DTYPE_FLOAT32:
            s_min = (scalar_t)min.v.f32;
            s_max = (scalar_t)max.v.f32;
            break;
        case DTYPE_FLOAT64:
            s_min = (scalar_t)min.v.f64;
            s_max = (scalar_t)max.v.f64;
            break;
        case DTYPE_INT32:
            s_min = (scalar_t)min.v.i32;
            s_max = (scalar_t)max.v.i32;
            break;
        case DTYPE_INT64:
            s_min = (scalar_t)min.v.i64;
            s_max = (scalar_t)max.v.i64;
            break;
        case DTYPE_INT8:
            s_min = (scalar_t)min.v.i8;
            s_max = (scalar_t)max.v.i8;
            break;
        case DTYPE_BOOL:
            s_min = (scalar_t)min.v.b;
            s_max = (scalar_t)max.v.b;
            break;
        default:
            s_min = (scalar_t)0;
            s_max = (scalar_t)0;
            break;
        }

        scale_kernel<scalar_t><<<cuda_grid_1d(total_elements), 256>>>(
            values,
            (scalar_t *)base,
            total_elements,
            s_min,
            s_max);
    });
}

template <typename T>
__global__ void arange_kernel(T *output, size_t total_elements, double start, double step)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += (size_t)blockDim.x * gridDim.x)
    {
        output[index] = static_cast<T>(start + static_cast<double>(index) * step);
    }
}

extern "C" cudaError_t launch_arange_kernel(
    void *base,
    dtype_t dtype,
    size_t total_elements,
    double start,
    double step)
{
    if (total_elements == 0)
        return cudaSuccess;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            arange_kernel<float><<<grid, 256>>>((float *)base, total_elements, start, step);
            break;
        case DTYPE_FLOAT64:
            arange_kernel<double><<<grid, 256>>>((double *)base, total_elements, start, step);
            break;
        case DTYPE_INT8:
            arange_kernel<int8_t><<<grid, 256>>>((int8_t *)base, total_elements, start, step);
            break;
        case DTYPE_INT16:
            arange_kernel<int16_t><<<grid, 256>>>((int16_t *)base, total_elements, start, step);
            break;
        case DTYPE_INT32:
            arange_kernel<int32_t><<<grid, 256>>>((int32_t *)base, total_elements, start, step);
            break;
        case DTYPE_INT64:
            arange_kernel<int64_t><<<grid, 256>>>((int64_t *)base, total_elements, start, step);
            break;
        case DTYPE_UINT8:
            arange_kernel<uint8_t><<<grid, 256>>>((uint8_t *)base, total_elements, start, step);
            break;
        case DTYPE_UINT16:
            arange_kernel<uint16_t><<<grid, 256>>>((uint16_t *)base, total_elements, start, step);
            break;
        case DTYPE_UINT32:
            arange_kernel<uint32_t><<<grid, 256>>>((uint32_t *)base, total_elements, start, step);
            break;
        case DTYPE_UINT64:
            arange_kernel<uint64_t><<<grid, 256>>>((uint64_t *)base, total_elements, start, step);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    return cudaDeviceSynchronize();
}

template <typename T>
__global__ void linspace_kernel(
    T *output,
    size_t total_elements,
    double start,
    double stop,
    bool endpoint)
{
    if (total_elements == 0)
        return;

    size_t denominator = endpoint ? total_elements - 1 : total_elements;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += (size_t)blockDim.x * gridDim.x)
    {
        if (endpoint && total_elements > 1 && index == total_elements - 1)
        {
            output[index] = static_cast<T>(stop);
            continue;
        }

        if (total_elements == 1)
        {
            output[index] = static_cast<T>(start);
            continue;
        }

        double fraction = static_cast<double>(index) / static_cast<double>(denominator);
        double value = start * (1.0 - fraction) + stop * fraction;
        output[index] = static_cast<T>(value);
    }
}

extern "C" cudaError_t launch_linspace_kernel(
    void *base,
    dtype_t dtype,
    size_t total_elements,
    double start,
    double stop,
    bool endpoint)
{
    if (total_elements == 0)
        return cudaSuccess;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            linspace_kernel<float><<<grid, 256>>>((float *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_FLOAT64:
            linspace_kernel<double><<<grid, 256>>>((double *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_INT8:
            linspace_kernel<int8_t><<<grid, 256>>>((int8_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_INT16:
            linspace_kernel<int16_t><<<grid, 256>>>((int16_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_INT32:
            linspace_kernel<int32_t><<<grid, 256>>>((int32_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_INT64:
            linspace_kernel<int64_t><<<grid, 256>>>((int64_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_UINT8:
            linspace_kernel<uint8_t><<<grid, 256>>>((uint8_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_UINT16:
            linspace_kernel<uint16_t><<<grid, 256>>>((uint16_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_UINT32:
            linspace_kernel<uint32_t><<<grid, 256>>>((uint32_t *)base, total_elements, start, stop, endpoint);
            break;
        case DTYPE_UINT64:
            linspace_kernel<uint64_t><<<grid, 256>>>((uint64_t *)base, total_elements, start, stop, endpoint);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    return cudaDeviceSynchronize();
}

template <typename T>
__global__ void logspace_kernel(
    T *output,
    size_t total_elements,
    double start,
    double stop,
    double base_value,
    bool endpoint,
    bool integer_dtype)
{
    if (total_elements == 0)
        return;

    size_t denominator = endpoint ? total_elements - 1 : total_elements;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += (size_t)blockDim.x * gridDim.x)
    {
        double exponent;
        if (total_elements == 1)
        {
            exponent = start;
        }
        else if (endpoint && index == total_elements - 1)
        {
            exponent = stop;
        }
        else
        {
            double fraction = static_cast<double>(index) / static_cast<double>(denominator);
            exponent = start * (1.0 - fraction) + stop * fraction;
        }

        double value = pow(base_value, exponent);
        if (integer_dtype)
        {
            // pow() may return an integer target one or two ulps below it.
            // Snap only values within a double-rounding tolerance; fractional
            // results such as 3.162... remain truncating conversions.
            double nearest = nearbyint(value);
            double scale = fabs(value) > 1.0 ? fabs(value) : 1.0;
            double tolerance = 8.0 * 2.2204460492503131e-16 * scale;
            if (fabs(value - nearest) <= tolerance)
                value = nearest;
        }
        output[index] = static_cast<T>(value);
    }
}

extern "C" cudaError_t launch_logspace_kernel(
    void *base,
    dtype_t dtype,
    size_t total_elements,
    double start,
    double stop,
    double base_value,
    bool endpoint)
{
    if (total_elements == 0)
        return cudaSuccess;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            logspace_kernel<float><<<grid, 256>>>((float *)base, total_elements, start, stop, base_value, endpoint, false);
            break;
        case DTYPE_FLOAT64:
            logspace_kernel<double><<<grid, 256>>>((double *)base, total_elements, start, stop, base_value, endpoint, false);
            break;
        case DTYPE_INT8:
            logspace_kernel<int8_t><<<grid, 256>>>((int8_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_INT16:
            logspace_kernel<int16_t><<<grid, 256>>>((int16_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_INT32:
            logspace_kernel<int32_t><<<grid, 256>>>((int32_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_INT64:
            logspace_kernel<int64_t><<<grid, 256>>>((int64_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_UINT8:
            logspace_kernel<uint8_t><<<grid, 256>>>((uint8_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_UINT16:
            logspace_kernel<uint16_t><<<grid, 256>>>((uint16_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_UINT32:
            logspace_kernel<uint32_t><<<grid, 256>>>((uint32_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        case DTYPE_UINT64:
            logspace_kernel<uint64_t><<<grid, 256>>>((uint64_t *)base, total_elements, start, stop, base_value, endpoint, true);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    return cudaDeviceSynchronize();
}

template <typename T>
__global__ void eye_kernel(
    T *output,
    size_t total_elements,
    int rows,
    int cols,
    int64_t diagonal_offset)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += (size_t)blockDim.x * gridDim.x)
    {
        int64_t row = (int64_t)(index / (size_t)cols);
        int64_t column = (int64_t)(index % (size_t)cols);
        bool on_diagonal;

        if (diagonal_offset >= 0)
        {
            on_diagonal = column >= row &&
                (uint64_t)(column - row) == (uint64_t)diagonal_offset;
        }
        else
        {
            uint64_t negative_offset = (uint64_t)(-(diagonal_offset + 1)) + 1;
            on_diagonal = row >= column &&
                (uint64_t)(row - column) == negative_offset;
        }

        output[index] = on_diagonal ? static_cast<T>(1) : static_cast<T>(0);
    }
}

extern "C" cudaError_t launch_eye_kernel(
    void *base,
    dtype_t dtype,
    size_t total_elements,
    int rows,
    int cols,
    int64_t offset)
{
    if (total_elements == 0)
        return cudaSuccess;
    if (rows <= 0 || cols <= 0)
        return cudaErrorInvalidValue;

    dim3 grid = cuda_grid_1d(total_elements);
    switch (dtype)
    {
        case DTYPE_FLOAT32:
            eye_kernel<float><<<grid, 256>>>((float *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_FLOAT64:
            eye_kernel<double><<<grid, 256>>>((double *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_INT8:
            eye_kernel<int8_t><<<grid, 256>>>((int8_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_INT16:
            eye_kernel<int16_t><<<grid, 256>>>((int16_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_INT32:
            eye_kernel<int32_t><<<grid, 256>>>((int32_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_INT64:
            eye_kernel<int64_t><<<grid, 256>>>((int64_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_UINT8:
            eye_kernel<uint8_t><<<grid, 256>>>((uint8_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_UINT16:
            eye_kernel<uint16_t><<<grid, 256>>>((uint16_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_UINT32:
            eye_kernel<uint32_t><<<grid, 256>>>((uint32_t *)base, total_elements, rows, cols, offset);
            break;
        case DTYPE_UINT64:
            eye_kernel<uint64_t><<<grid, 256>>>((uint64_t *)base, total_elements, rows, cols, offset);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    return cudaDeviceSynchronize();
}
