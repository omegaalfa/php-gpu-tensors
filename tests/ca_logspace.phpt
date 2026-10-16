--TEST--
CudaArray logspace creates logarithmically spaced GPU sequences
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

var_dump(CudaArray::logspace(0, 3, 4, true, 10.0, 'float64')->toArray());
var_dump(CudaArray::logspace(0, 2, 3, false)->toArray());
var_dump(CudaArray::logspace(2, 0, 3, true, 2.0, 'float64')->toArray());
var_dump(CudaArray::logspace(0, -400, 2, true, 10.0, 'float64')->toArray());
var_dump(CudaArray::logspace(2, 9, 1)->toArray());
var_dump(CudaArray::logspace(0, 3, 0)->getShape());
var_dump(CudaArray::logspace(0, 3, 4, true, 10.0, 'int32')->toArray());

$float32 = CudaArray::logspace(0, 3, 4, true, 10.0, 'float32')->toArray();
$float64 = CudaArray::logspace(0, 3, 4, true, 10.0, 'float64')->toArray();
if (abs($float32[2] - 100.0) > 1.0e-4 || $float32[3] !== 1000.0 ||
    abs($float64[2] - 100.0) > 1.0e-12 || $float64[3] !== 1000.0) {
    echo "precision mismatch\n";
    exit(1);
}

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $values = CudaArray::logspace(0, 2, 3, true, 10.0, $dtype)->toArray();
    if (count($values) !== 3 || (float)$values[0] !== 1.0 || (float)$values[2] !== 100.0) {
        printf("dtype %s mismatch\n", $dtype);
        exit(1);
    }
}

// Same exponent interpolation used by ZMatrix for compatible finite inputs.
$reference = [];
for ($i = 0; $i < 4; $i++) {
    $exponent = 1.0 + $i * (4.0 - 1.0) / (4 - 1);
    $reference[] = 3.0 ** $exponent;
}
$actual = CudaArray::logspace(1, 4, 4, true, 3.0, 'float64')->toArray();
foreach ($reference as $i => $expected) {
    if (abs($actual[$i] - $expected) > 1.0e-12) {
        echo "reference mismatch\n";
        exit(1);
    }
}

$uint8Limit = CudaArray::logspace(0, log10(255), 2, true, 10.0, 'uint8')->toArray();
$int8Limit = CudaArray::logspace(0, log10(127), 2, true, 10.0, 'int8')->toArray();
$int32Limit = CudaArray::logspace(0, log10(2147483647), 2, true, 10.0, 'int32')->toArray();
if ($uint8Limit !== [1, 255] || $int8Limit !== [1, 127] || $int32Limit !== [1, 2147483647]) {
    echo "integer endpoint mismatch\n";
    exit(1);
}

$fractionalInteger = CudaArray::logspace(0, 1, 2, false, 10.0, 'int32')->toArray();
if ($fractionalInteger !== [1, 3]) {
    echo "fractional integer conversion mismatch\n";
    exit(1);
}
echo "logspace checks passed\n";
?>
--EXPECT--
array(4) {
  [0]=>
  float(1)
  [1]=>
  float(10)
  [2]=>
  float(100)
  [3]=>
  float(1000)
}
array(3) {
  [0]=>
  float(1)
  [1]=>
  float(4.6415886878967285)
  [2]=>
  float(21.544347763061523)
}
array(3) {
  [0]=>
  float(4)
  [1]=>
  float(2)
  [2]=>
  float(1)
}
array(2) {
  [0]=>
  float(1)
  [1]=>
  float(0)
}
array(1) {
  [0]=>
  float(100)
}
array(1) {
  [0]=>
  int(0)
}
array(4) {
  [0]=>
  int(1)
  [1]=>
  int(10)
  [2]=>
  int(100)
  [3]=>
  int(1000)
}
logspace checks passed
