--TEST--
CudaArray identity reuses the eye diagonal factory
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

if (CudaArray::identity(3)->toArray() != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    echo "default mismatch\n";
    exit(1);
}
if (CudaArray::identity(2, 'float64')->getShape() != [2, 2]) {
    echo "shape mismatch\n";
    exit(1);
}
if (CudaArray::identity(0)->getShape() != [0, 0]) {
    echo "empty shape mismatch\n";
    exit(1);
}

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $identity = CudaArray::identity(2, $dtype)->toArray();
    $eye = CudaArray::eye(2, 2, 0, $dtype)->toArray();
    if ($identity != [[1, 0], [0, 1]] || $identity != $eye) {
        printf("dtype %s mismatch\n", $dtype);
        exit(1);
    }
}
echo "identity checks passed\n";
?>
--EXPECT--
identity checks passed
