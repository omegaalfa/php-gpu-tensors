--TEST--
CudaArray logspace validates bases, ranges, and limits
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::logspace(0, 1, 3, true, 0.0),
    static fn() => CudaArray::logspace(0, 1, 3, true, -2.0),
    static fn() => CudaArray::logspace(0, 1, 3, true, NAN),
    static fn() => CudaArray::logspace(0, 1, 3, true, INF),
    static fn() => CudaArray::logspace(0, 1, -1),
    static fn() => CudaArray::logspace(0, 1, PHP_INT_MAX),
    static fn() => CudaArray::logspace(0, 400, 2, true, 10.0, 'float64'),
    static fn() => CudaArray::logspace(0, 3, 4, true, 10.0, 'bool'),
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
