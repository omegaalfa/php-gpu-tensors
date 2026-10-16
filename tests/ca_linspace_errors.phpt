--TEST--
CudaArray linspace validates arguments and limits
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::linspace(0, 1, -1),
    static fn() => CudaArray::linspace(NAN, 1),
    static fn() => CudaArray::linspace(0, INF),
    static fn() => CudaArray::linspace(0, 1, PHP_INT_MAX),
    static fn() => CudaArray::linspace(0, 1, 2, true, 'bool'),
    static fn() => CudaArray::linspace(0, 300, 2, true, 'uint8'),
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
