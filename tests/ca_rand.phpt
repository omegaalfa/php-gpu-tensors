--TEST--
CudaArray rand generates non-repeating uniform GPU samples
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

$first = CudaArray::rand([2, 3, 4], -2.5, 7.5, 'float32')->toArray();
$second = CudaArray::rand([2, 3, 4], -2.5, 7.5, 'float32')->toArray();
if ($first === $second) {
    echo "repeated float32 sequence\n";
    exit(1);
}
foreach ($first as $row) {
    foreach ($row as $column) {
        foreach ($column as $value) {
            if (!is_finite($value) || $value < -2.5 || $value >= 7.5) {
                echo "float32 interval mismatch\n";
                exit(1);
            }
        }
    }
}

$double = CudaArray::rand([256], 0.25, 0.5, 'float64')->toArray();
$sum = 0.0;
foreach ($double as $value) {
    if (!is_finite($value) || $value < 0.25 || $value >= 0.5) {
        echo "float64 interval mismatch\n";
        exit(1);
    }
    $sum += $value;
}
if (abs(($sum / count($double)) - 0.375) > 0.06) {
    echo "distribution mismatch\n";
    exit(1);
}

echo "rand checks passed\n";
?>
--EXPECT--
rand checks passed
