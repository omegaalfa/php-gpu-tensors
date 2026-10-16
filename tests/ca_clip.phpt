--TEST--
CudaArray clip clamps GPU tensors across dtypes and layouts
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_clip(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

function clip_zero_sign(CudaArray $tensor, string $dtype, int $index): bool {
    $bytes = $tensor->toBuffer();
    if ($dtype === 'float32') {
        $word = unpack('V', substr($bytes, $index * 4, 4))[1];
        return (bool) ($word & 0x80000000);
    }
    $words = unpack('Vlow/Vhigh', substr($bytes, $index * 8, 8));
    return (bool) ($words['high'] & 0x80000000);
}

$input = new CudaArray([-3.5, -1.0, 0.5, 4.0], 'float64');
$clipped = $input->clip(-1.0, 2.0);
check_clip($clipped->toArray() === [-1.0, -1.0, 0.5, 2.0], 'basic clamp');
check_clip($clipped->getShape() === [4] && $clipped->dtype() === 'float64', 'metadata');

$equal = (new CudaArray([-2, 0, 3], 'int32'))->clip(1, 1);
check_clip($equal->toArray() === [1, 1, 1], 'equal bounds');

$unsigned = (new CudaArray([0, 10, 255], 'uint8'))->clip(5, 200);
check_clip($unsigned->toArray() === [5, 10, 200] && $unsigned->dtype() === 'uint8', 'unsigned');

$int = (new CudaArray([-100, -2, 7, 100], 'int8'))->clip(-10, 10);
check_clip($int->toArray() === [-10, -2, 7, 10], 'integer');

$view = (new CudaArray([[1, 9, 3], [8, 2, 7]], 'int32'))->slice(':, ::2');
$viewResult = $view->clip(2, 6);
check_clip($viewResult->toArray() === [[2, 3], [6, 6]], 'strided view');

$special = new CudaArray([NAN, -INF, INF], 'float32')->clip(-1.0, 1.0)->toArray();
check_clip(is_nan($special[0]), 'NaN preserved');
check_clip($special[1] === -1.0 && $special[2] === 1.0, 'infinity clamp');

foreach (['float32', 'float64'] as $dtype) {
    $positive = (new CudaArray([-0.0], $dtype))->clip(0.0, 1.0);
    $negative = (new CudaArray([0.0], $dtype))->clip(-1.0, -0.0);
    check_clip(!clip_zero_sign($positive, $dtype, 0), "$dtype lower positive zero");
    check_clip(clip_zero_sign($negative, $dtype, 0), "$dtype upper negative zero");
}

$empty = CudaArray::zeros([0, 3], 'float32')->clip(-1.0, 1.0);
check_clip($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty tensor');

$independent = $input->clip(-1.0, 2.0);
$input[0] = 99.0;
check_clip($independent->toArray()[0] === -1.0, 'independent result');

foreach ([
    [-2.0, -3.0],
    [NAN, 1.0],
] as [$minimum, $maximum]) {
    try {
        (new CudaArray([1.0], 'float32'))->clip($minimum, $maximum);
        throw new RuntimeException('invalid clip accepted');
    } catch (Cuda\InvalidArgumentException $error) {
    }
}

try {
    (new CudaArray([1], 'int32'))->clip(0.5, 1.5);
    throw new RuntimeException('fractional integer bounds accepted');
} catch (Cuda\InvalidArgumentException $error) {
}

try {
    (new CudaArray([1.0], 'float32'))->clip(-INF, INF);
    echo "infinite bounds accepted\n";
} catch (Throwable $error) {
    throw new RuntimeException('infinite float bounds rejected');
}

try {
    (new CudaArray([1], 'uint8'))->clip(-1, 300);
    throw new RuntimeException('out of range integer bounds accepted');
} catch (Cuda\InvalidArgumentException $error) {
}

echo "clip checked\n";
?>
--EXPECT--
infinite bounds accepted
clip checked
