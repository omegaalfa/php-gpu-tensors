--TEST--
CudaArray tile repeats tensors and materializes strided views on the GPU
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_tile(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$one = new CudaArray([1, 2, 3], 'float64');
check_tile($one->tile(1)->toArray() === [1.0, 2.0, 3.0], 'factor one');
check_tile($one->tile(3)->toArray() === [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], '1D repeat');
check_tile($one->tile(3)->getShape() === [9], '1D shape');

$matrix = new CudaArray([[1, 2], [3, 4]], 'int32');
$tiled = $matrix->tile(2);
check_tile($tiled->getShape() === [4, 2], '2D shape');
check_tile($tiled->toArray() === [[1, 2], [3, 4], [1, 2], [3, 4]], '2D repeat');
check_tile($tiled->dtype() === 'int32', 'dtype');

$view = $matrix->slice(':, ::2');
$viewTiled = $view->tile(2);
check_tile($viewTiled->toArray() === [[1], [3], [1], [3]], 'strided view');
$matrix[0][0] = 99;
check_tile($viewTiled->toArray() === [[1], [3], [1], [3]], 'independent storage');

$three = new CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 'uint16');
check_tile($three->tile(2)->toArray() === [
    [[1, 2], [3, 4]], [[5, 6], [7, 8]],
    [[1, 2], [3, 4]], [[5, 6], [7, 8]],
], '3D repeat');

$empty = CudaArray::zeros([0, 2], 'float32')->tile(2);
check_tile($empty->getShape() === [0, 2] && $empty->getSize() === 0, 'empty shape');
check_tile($empty->toArray() === [], 'empty values');

foreach ([0, -1] as $times) {
    try {
        $one->tile($times);
        throw new RuntimeException('invalid factor accepted');
    } catch (Cuda\InvalidArgumentException $e) {
    }
}
try {
    (new CudaArray([1]))->slice('0')->tile(2);
    throw new RuntimeException('scalar accepted');
} catch (Cuda\InvalidArgumentException $e) {
}
echo "tile checked\n";
?>
--EXPECT--
tile checked
