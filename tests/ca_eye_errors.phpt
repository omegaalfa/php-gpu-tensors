--TEST--
CudaArray eye validates dimensions, allocation limits, and dtypes
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::eye(-1),
    static fn() => CudaArray::eye(2, -1),
    static fn() => CudaArray::eye(PHP_INT_MAX),
    static fn() => CudaArray::eye(2, 2, 0, 'bool'),
    static fn() => CudaArray::eye(2, 2, 0, 'not-a-dtype'),
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
