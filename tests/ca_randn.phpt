--TEST--
CudaArray randn generates normal GPU samples
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$matrix = CudaArray::randn([2, 3], 2.0, 0.5, 'float64');
if ($matrix->getShape() !== [2, 3] || $matrix->getSize() !== 6) {
    echo "shape mismatch\n";
    exit(1);
}

$constant = CudaArray::randn([2, 2], 3.25, 0.0, 'float64')->toArray();
foreach ($constant as $row) {
    foreach ($row as $value) {
        if ($value !== 3.25) {
            echo "zero std mismatch\n";
            exit(1);
        }
    }
}

if (CudaArray::randn([0, 4], 0.0, 1.0, 'float32')->getSize() !== 0) {
    echo "empty mismatch\n";
    exit(1);
}

$samples = CudaArray::randn([20000], 2.0, 0.75, 'float32')->toArray();
$sum = array_sum($samples);
$mean = $sum / count($samples);
$variance = 0.0;
foreach ($samples as $value) {
    $variance += ($value - $mean) ** 2;
}
$std = sqrt($variance / count($samples));
if (abs($mean - 2.0) > 0.06 || abs($std - 0.75) > 0.06) {
    printf("statistics mismatch: mean=%.6f std=%.6f\n", $mean, $std);
    exit(1);
}

$double_samples = CudaArray::randn([2000], -1.0, 1.25, 'float64')->toArray();
if (count($double_samples) !== 2000) {
    echo "float64 mismatch\n";
    exit(1);
}
echo "randn checks passed\n";
?>
--EXPECT--
randn checks passed
