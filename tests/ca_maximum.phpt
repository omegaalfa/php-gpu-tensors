--TEST--
CudaArray maximum with broadcasting, promotion, views and special floats
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_maximum(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

function has_zero_sign_max(CudaArray $tensor, string $dtype, int $index, bool $negative): bool {
    $bytes = $tensor->toBuffer();
    if ($dtype === 'float32') {
        $word = unpack('V', substr($bytes, $index * 4, 4))[1];
        $isNegative = (bool) ($word & 0x80000000);
    } else {
        $words = unpack('Vlow/Vhigh', substr($bytes, $index * 8, 8));
        $isNegative = (bool) ($words['high'] & 0x80000000);
    }
    return $isNegative === $negative;
}

$a = new CudaArray([[3, -2, 7], [4, 5, -6]], 'int32');
$b = new CudaArray([[1, -3, 8], [9, 2, -7]], 'int32');
$same = CudaArray::maximum($a, $b);
check_maximum($same->toArray() === [[3, -2, 8], [9, 5, -6]], 'same shape');
check_maximum($same->getShape() === [2, 3] && $same->dtype() === 'int32', 'same metadata');

$row = new CudaArray([[2, 8, -1]], 'float64');
$broadcast = CudaArray::maximum($a, $row);
check_maximum($broadcast->toArray() === [[3.0, 8.0, 7.0], [4.0, 8.0, -1.0]], 'row broadcast');
check_maximum($broadcast->dtype() === 'float64', 'promotion');

$column = new CudaArray([[0], [10]], 'int32');
$multi = CudaArray::maximum($a, $column);
check_maximum($multi->toArray() === [[3, 0, 7], [10, 10, 10]], 'column broadcast');

$mixed = CudaArray::maximum(new CudaArray([1, 250], 'uint8'), new CudaArray([2, 3], 'int32'));
check_maximum($mixed->toArray() === [2, 250] && $mixed->dtype() === 'int32', 'mixed dtype');

$view = (new CudaArray([[9, 1], [8, 2]], 'int32'))->transpose();
$viewResult = CudaArray::maximum($view, new CudaArray([[3, 4], [5, 0]], 'int32'));
check_maximum($viewResult->toArray() === [[9, 8], [5, 2]], 'strided view');

$nan = new CudaArray([NAN, -INF, INF], 'float64');
$special = CudaArray::maximum($nan, new CudaArray([1.0, INF, -INF], 'float64'))->toArray();
check_maximum(is_nan($special[0]), 'NaN propagation');
check_maximum($special[1] === INF && $special[2] === INF, 'infinity ordering');

foreach (['float32', 'float64'] as $zeroDtype) {
    $positive = new CudaArray([0.0], $zeroDtype);
    $negative = new CudaArray([-0.0], $zeroDtype);
    check_maximum(has_zero_sign_max(CudaArray::maximum($positive, $positive), $zeroDtype, 0, false), "$zeroDtype +0,+0");
    check_maximum(has_zero_sign_max(CudaArray::maximum($negative, $negative), $zeroDtype, 0, true), "$zeroDtype -0,-0");
    check_maximum(has_zero_sign_max(CudaArray::maximum($positive, $negative), $zeroDtype, 0, false), "$zeroDtype +0,-0");
    check_maximum(has_zero_sign_max(CudaArray::maximum($negative, $positive), $zeroDtype, 0, false), "$zeroDtype -0,+0");
}

$zeroBroadcast = CudaArray::maximum(
    new CudaArray([[0.0], [-0.0]], 'float64'),
    new CudaArray([[0.0, -0.0]], 'float64')
);
check_maximum(has_zero_sign_max($zeroBroadcast, 'float64', 0, false), 'broadcast +0,+0');
check_maximum(has_zero_sign_max($zeroBroadcast, 'float64', 1, false), 'broadcast +0,-0');
check_maximum(has_zero_sign_max($zeroBroadcast, 'float64', 2, false), 'broadcast -0,+0');
check_maximum(has_zero_sign_max($zeroBroadcast, 'float64', 3, true), 'broadcast -0,-0');

$empty = CudaArray::maximum(CudaArray::zeros([0, 3], 'float32'), CudaArray::zeros([1, 3], 'float32'));
check_maximum($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty broadcast');

$independent = CudaArray::maximum($a, $b);
$a[0][0] = -99;
check_maximum($independent->toArray()[0][0] === 3, 'independent result');

try {
    CudaArray::maximum(new CudaArray([1, 2], 'int32'), new CudaArray([1, 2, 3], 'int32'));
    throw new RuntimeException('invalid maximum accepted');
} catch (Cuda\InvalidArgumentException $error) {
}

echo "maximum checked\n";
?>
--EXPECT--
maximum checked
