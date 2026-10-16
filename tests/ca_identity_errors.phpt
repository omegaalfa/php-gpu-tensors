--TEST--
CudaArray identity validates dimensions, allocation limits, and dtypes
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::identity(-1),
    static fn() => CudaArray::identity(PHP_INT_MAX),
    static fn() => CudaArray::identity(2, 'bool'),
    static fn() => CudaArray::identity(2, 'not-a-dtype'),
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
