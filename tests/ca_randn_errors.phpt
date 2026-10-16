--TEST--
CudaArray randn validates shape, parameters, and dtypes
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::randn([]),
    static fn() => CudaArray::randn([-1]),
    static fn() => CudaArray::randn(['2']),
    static fn() => CudaArray::randn([PHP_INT_MAX, 2]),
    static fn() => CudaArray::randn([2], NAN),
    static fn() => CudaArray::randn([2], 0.0, INF),
    static fn() => CudaArray::randn([2], 0.0, -1.0),
    static fn() => CudaArray::randn([2], 0.0, 1.0, 'int32'),
    static fn() => CudaArray::randn([2], 1.0e39, 1.0, 'float32'),
];

foreach ($cases as $case) {
    try {
        $case();
        echo "NOT_THROWN\n";
    } catch (Throwable $exception) {
        echo "THROWN\n";
    }
}
?>
--EXPECT--
THROWN
THROWN
THROWN
THROWN
THROWN
THROWN
THROWN
THROWN
THROWN
