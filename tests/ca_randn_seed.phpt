--TEST--
CudaArray randn avoids repeated consecutive sequences
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension not loaded');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$first = CudaArray::randn([64], 0.0, 1.0, 'float32')->toArray();
$second = CudaArray::randn([64], 0.0, 1.0, 'float32')->toArray();
if ($first === $second) {
    echo "repeated float32 sequence\n";
    exit(1);
}

$first = CudaArray::randn([65], 0.0, 1.0, 'float64')->toArray();
$second = CudaArray::randn([65], 0.0, 1.0, 'float64')->toArray();
if ($first === $second) {
    echo "repeated float64 sequence\n";
    exit(1);
}

$constant_a = CudaArray::randn([8], 4.0, 0.0, 'float64')->toArray();
$constant_b = CudaArray::randn([8], 4.0, 0.0, 'float64')->toArray();
if ($constant_a !== $constant_b) {
    echo "std zero mismatch\n";
    exit(1);
}

echo "randn seed checks passed\n";
?>
--EXPECT--
randn seed checks passed
