--TEST--
CudaArray randn rejects non-finite transformed values
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$cases = [
    static fn() => CudaArray::randn([64], 3.402823466e38, 3.402823466e38, 'float32'),
    static fn() => CudaArray::randn([64], 1.7976931348623157e308, 1.7976931348623157e308, 'float64'),
];

foreach ($cases as $case) {
    try {
        $case();
        echo "NOT_THROWN\n";
    } catch (Throwable $exception) {
        echo "THROWN\n";
    }
}

$finite = CudaArray::randn([64], 1.0, 0.25, 'float64')->toArray();
foreach ($finite as $value) {
    if (!is_finite($value)) {
        echo "NON_FINITE\n";
        exit(1);
    }
}

$near_float32 = CudaArray::randn([1], 3.402823466e38, 0.0, 'float32')->toArray();
$near_float64 = CudaArray::randn([1], 1.7976931348623157e308, 0.0, 'float64')->toArray();
if (!is_finite($near_float32[0]) || !is_finite($near_float64[0])) {
    echo "FINITE_LIMIT_MISMATCH\n";
    exit(1);
}

$constant = CudaArray::randn([4], 3.5, 0.0, 'float32')->toArray();
if ($constant !== [3.5, 3.5, 3.5, 3.5]) {
    echo "STD_ZERO_MISMATCH\n";
    exit(1);
}

echo "randn overflow checks passed\n";
?>
--EXPECT--
THROWN
THROWN
randn overflow checks passed
