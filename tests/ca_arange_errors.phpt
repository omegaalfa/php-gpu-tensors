--TEST--
CudaArray arange validates arguments and dtype ranges
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::arange(5, null),
    static fn() => CudaArray::arange(0, 5, null),
    static fn() => CudaArray::arange(0, 5, 0),
    static fn() => CudaArray::arange(NAN, 5, 1),
    static fn() => CudaArray::arange(0, INF, 1),
    static fn() => CudaArray::arange(0, 5, 1, 'not-a-dtype'),
    static fn() => CudaArray::arange(0, 5, 1, 'bool'),
    static fn() => CudaArray::arange(0, PHP_INT_MAX, 0.000000000000001),
    static fn() => CudaArray::arange(250, 260, 1, 'uint8'),
    static fn() => CudaArray::arange('0', 5, 1),
];

foreach ($cases as $case) {
    try {
        $case();
        echo "UNEXPECTED\n";
    } catch (Throwable $e) {
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
