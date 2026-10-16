--TEST--
CudaArray eye creates diagonal GPU matrices
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

var_dump(CudaArray::eye(3)->toArray());
var_dump(CudaArray::eye(2, 3, 1)->toArray());
var_dump(CudaArray::eye(3, 3, -1)->toArray());
var_dump(CudaArray::eye(2, 4, 0, 'float64')->getShape());
var_dump(CudaArray::eye(0, 3)->getSize());
var_dump(CudaArray::eye(3, 0)->toArray());
var_dump(CudaArray::eye(3, 3, 9)->toArray());

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $values = CudaArray::eye(2, 2, 0, $dtype)->toArray();
    if ($values != [[1, 0], [0, 1]]) {
        printf("dtype %s mismatch\n", $dtype);
        exit(1);
    }
}

// Same diagonal rule used by ZMatrix for compatible finite dimensions.
$reference = [[0, 1, 0], [0, 0, 1]];
$actual = CudaArray::eye(2, 3, 1, 'int32')->toArray();
if ($actual != $reference) {
    echo "reference mismatch\n";
    exit(1);
}
echo "eye checks passed\n";
?>
--EXPECT--
array(3) {
  [0]=>
  array(3) {
    [0]=>
    float(1)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
  [1]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(1)
    [2]=>
    float(0)
  }
  [2]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(1)
  }
}
array(2) {
  [0]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(1)
    [2]=>
    float(0)
  }
  [1]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(1)
  }
}
array(3) {
  [0]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
  [1]=>
  array(3) {
    [0]=>
    float(1)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
  [2]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(1)
    [2]=>
    float(0)
  }
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(4)
}
int(0)
array(3) {
  [0]=>
  array(0) {
  }
  [1]=>
  array(0) {
  }
  [2]=>
  array(0) {
  }
}
array(3) {
  [0]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
  [1]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
  [2]=>
  array(3) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(0)
  }
}
eye checks passed
