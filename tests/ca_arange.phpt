--TEST--
CudaArray arange creates exclusive GPU sequences
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

var_dump(CudaArray::arange(0, 5, 1)->toArray());
var_dump(CudaArray::arange(5, 0, -1)->toArray());
var_dump(CudaArray::arange(0, 1, 0.25)->toArray());
var_dump(CudaArray::arange(5, 0, 1)->getShape());
var_dump(CudaArray::arange(5, 6, 2)->toArray());
var_dump(CudaArray::arange(-2, 2, 1, 'int32')->dtype());
var_dump(CudaArray::arange(0, 3, 1, 'float64')->toArray());
var_dump(CudaArray::arange(5)->toArray());

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $values = CudaArray::arange(0, 3, 1, $dtype)->toArray();
    if (array_map('floatval', $values) !== [0.0, 1.0, 2.0]) {
        printf("dtype %s mismatch\n", $dtype);
        exit(1);
    }
}
echo "all numeric dtypes passed\n";
?>
--EXPECT--
array(5) {
  [0]=>
  float(0)
  [1]=>
  float(1)
  [2]=>
  float(2)
  [3]=>
  float(3)
  [4]=>
  float(4)
}
array(5) {
  [0]=>
  float(5)
  [1]=>
  float(4)
  [2]=>
  float(3)
  [3]=>
  float(2)
  [4]=>
  float(1)
}
array(4) {
  [0]=>
  float(0)
  [1]=>
  float(0.25)
  [2]=>
  float(0.5)
  [3]=>
  float(0.75)
}
array(1) {
  [0]=>
  int(0)
}
array(1) {
  [0]=>
  float(5)
}
string(5) "int32"
array(3) {
  [0]=>
  float(0)
  [1]=>
  float(1)
  [2]=>
  float(2)
}
array(5) {
  [0]=>
  float(0)
  [1]=>
  float(1)
  [2]=>
  float(2)
  [3]=>
  float(3)
  [4]=>
  float(4)
}
all numeric dtypes passed
