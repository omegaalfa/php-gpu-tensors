--TEST--
CudaArray copy preserves logical values and owns independent contiguous storage
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_copy(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$source = new CudaArray([[0, 1, 2, 3], [4, 5, 6, 7]], 'float64');
$copy = $source->copy();
check_copy($copy !== $source, 'new object');
check_copy($copy->getShape() === [2, 4], 'shape');
check_copy($copy->dtype() === 'float64', 'dtype');
check_copy($copy->getStrides() === [4, 1], 'contiguous strides');
check_copy($copy->toArray() === [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 'values');

$source[0][0] = 99;
check_copy($copy->toArray()[0][0] === 0.0, 'independent storage');

$view = $source->slice('::2, 1::2');
$viewCopy = $view->copy();
check_copy($viewCopy->getShape() === [1, 2], 'view shape');
check_copy($viewCopy->getStrides() === [2, 1], 'view materialized strides');
check_copy($viewCopy->toArray() === [[1.0, 3.0]], 'strided view values');
$source[0][1] = 88;
check_copy($viewCopy->toArray() === [[1.0, 3.0]], 'independent view copy');

$three = new CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 'int32');
$threeCopy = $three->copy();
check_copy($threeCopy->getShape() === [2, 2, 2], '3D shape');
check_copy($threeCopy->dtype() === 'int32', '3D dtype');
check_copy($threeCopy->toArray() === [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], '3D values');

$empty = CudaArray::zeros([0, 3], 'float32')->copy();
check_copy($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty metadata');
check_copy($empty->toArray() === [], 'empty values');

unset($source, $view, $three);
check_copy($copy->toArray() === [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 'source lifetime');
echo "copy checked\n";
?>
--EXPECT--
copy checked
