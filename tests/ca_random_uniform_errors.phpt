--TEST--
CudaArray randomUniform validates bounds, shape, and dtype
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::randomUniform([], 0.0, 1.0, 1),
    static fn() => CudaArray::randomUniform([-1], 0.0, 1.0, 1),
    static fn() => CudaArray::randomUniform(['2'], 0.0, 1.0, 1),
    static fn() => CudaArray::randomUniform([PHP_INT_MAX, 2], 0.0, 1.0, 1),
    static fn() => CudaArray::randomUniform([2], NAN, 1.0, 1),
    static fn() => CudaArray::randomUniform([2], 0.0, INF, 1),
    static fn() => CudaArray::randomUniform([2], 1.0, 1.0, 1),
    static fn() => CudaArray::randomUniform([2], 2.0, 1.0, 1),
    static fn() => CudaArray::randomUniform([2], 0.0, 1.0, 1, 'int32'),
    static fn() => CudaArray::randomUniform([2], 1.0e39, 1.0e40, 1, 'float32'),
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
THROWN
