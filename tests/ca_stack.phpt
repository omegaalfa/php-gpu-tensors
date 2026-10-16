--TEST--
CudaArray stack creates an independent leading dimension and handles views
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_stack(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$a = new CudaArray([1, 2, 3], 'float64');
$b = new CudaArray([4, 5, 6], 'float64');
$one = CudaArray::stack([$a, $b]);
check_stack($one->getShape() === [2, 3], '1D shape');
check_stack($one->toArray() === [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], '1D values');
check_stack($one->dtype() === 'float64', '1D dtype');

$m1 = new CudaArray([[1, 2], [3, 4]], 'int32');
$m2 = new CudaArray([[5, 6], [7, 8]], 'int32');
$matrix = CudaArray::stack([$m1, $m2]);
check_stack($matrix->getShape() === [2, 2, 2], '2D shape');
check_stack($matrix->toArray() === [
    [[1, 2], [3, 4]], [[5, 6], [7, 8]],
], '2D values');

$v1 = $m1->slice(':, ::2');
$v2 = $m2->slice(':, ::2');
$views = CudaArray::stack([$v1, $v2]);
check_stack($views->getShape() === [2, 2, 1], 'view shape');
check_stack($views->toArray() === [[[1], [3]], [[5], [7]]], 'view values');
$m1[0][0] = 99;
check_stack($views->toArray() === [[[1], [3]], [[5], [7]]], 'independent result');

$three = CudaArray::stack([
    new CudaArray([[[1, 2], [3, 4]]], 'uint16'),
    new CudaArray([[[5, 6], [7, 8]]], 'uint16'),
]);
check_stack($three->getShape() === [2, 1, 2, 2], '3D shape');
check_stack($three->toArray() === [
    [[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]],
], '3D values');

$empty = CudaArray::stack([
    CudaArray::zeros([0, 2], 'float32'),
    CudaArray::zeros([0, 2], 'float32'),
]);
check_stack($empty->getShape() === [2, 0, 2] && $empty->getSize() === 0, 'empty tensors');
check_stack($empty->toArray() === [[], []], 'empty values');

foreach ([[], [$a, new CudaArray([1, 2], 'float64')], [$a, new CudaArray([1, 2, 3], 'int32')], [$a, [1, 2, 3]]] as $invalid) {
    try {
        CudaArray::stack($invalid);
        throw new RuntimeException('invalid stack accepted');
    } catch (Cuda\InvalidArgumentException $e) {
    }
}
unset($a, $b, $m1, $m2, $v1, $v2);
check_stack($one->toArray() === [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'source lifetime');
echo "stack checked\n";
?>
--EXPECT--
stack checked
