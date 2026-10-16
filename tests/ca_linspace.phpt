--TEST--
CudaArray linspace creates uniformly spaced GPU sequences
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

var_dump(CudaArray::linspace(0, 1, 5)->toArray());
var_dump(CudaArray::linspace(0, 1, 5, false)->toArray());
var_dump(CudaArray::linspace(5, 1, 3, true, 'float64')->toArray());
var_dump(CudaArray::linspace(7, 9, 1)->toArray());
var_dump(CudaArray::linspace(7, 9, 0)->getShape());
var_dump(CudaArray::linspace(0, 4, 5, true, 'int32')->toArray());

$float32 = CudaArray::linspace(0, 1, 11, true, 'float32')->toArray();
$float64 = CudaArray::linspace(0, 1, 11, true, 'float64')->toArray();
if (abs($float32[3] - 0.3) > 1.0e-6 || $float32[10] !== 1.0 ||
    abs($float64[3] - 0.3) > 1.0e-15 || $float64[10] !== 1.0) {
    echo "precision mismatch\n";
    exit(1);
}

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $values = CudaArray::linspace(0, 4, 5, true, $dtype)->toArray();
    if (count($values) !== 5 || (float)$values[0] !== 0.0 || (float)$values[4] !== 4.0) {
        printf("dtype %s mismatch\n", $dtype);
        exit(1);
    }
}

// Same mathematical rule used by ZMatrix for compatible finite inputs.
$reference = [];
for ($i = 0; $i < 7; $i++) {
    $reference[] = 2.0 + $i * (8.0 - 2.0) / (7 - 1);
}
$actual = CudaArray::linspace(2, 8, 7, true, 'float64')->toArray();
foreach ($reference as $i => $expected) {
    if (abs($actual[$i] - $expected) > 1.0e-15) {
        echo "reference mismatch\n";
        exit(1);
    }
}
echo "linspace checks passed\n";
?>
--EXPECT--
array(5) {
  [0]=>
  float(0)
  [1]=>
  float(0.25)
  [2]=>
  float(0.5)
  [3]=>
  float(0.75)
  [4]=>
  float(1)
}
array(5) {
  [0]=>
  float(0)
  [1]=>
  float(0.20000000298023224)
  [2]=>
  float(0.4000000059604645)
  [3]=>
  float(0.6000000238418579)
  [4]=>
  float(0.800000011920929)
}
array(3) {
  [0]=>
  float(5)
  [1]=>
  float(3)
  [2]=>
  float(1)
}
array(1) {
  [0]=>
  float(7)
}
array(1) {
  [0]=>
  int(0)
}
array(5) {
  [0]=>
  int(0)
  [1]=>
  int(1)
  [2]=>
  int(2)
  [3]=>
  int(3)
  [4]=>
  int(4)
}
linspace checks passed
