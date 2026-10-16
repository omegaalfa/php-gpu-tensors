--TEST--
CudaArray minimum with broadcasting, promotion, views and special floats
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;

function check_minimum(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

function has_zero_sign(CudaArray $tensor, string $dtype, int $index, bool $negative): bool {
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
$same = CudaArray::minimum($a, $b);
check_minimum($same->toArray() === [[1, -3, 7], [4, 2, -7]], 'same shape');
check_minimum($same->getShape() === [2, 3] && $same->dtype() === 'int32', 'same metadata');

$row = new CudaArray([[2, 8, -1]], 'float64');
$broadcast = CudaArray::minimum($a, $row);
check_minimum($broadcast->toArray() === [[2.0, -2.0, -1.0], [2.0, 5.0, -6.0]], 'row broadcast');
check_minimum($broadcast->dtype() === 'float64', 'promotion');

$column = new CudaArray([[0], [10]], 'int32');
$multi = CudaArray::minimum($a, $column);
check_minimum($multi->toArray() === [[0, -2, 0], [4, 5, -6]], 'column broadcast');

$mixed = CudaArray::minimum(new CudaArray([1, 250], 'uint8'), new CudaArray([2, 3], 'int32'));
check_minimum($mixed->toArray() === [1, 3] && $mixed->dtype() === 'int32', 'mixed dtype');

$view = (new CudaArray([[9, 1], [8, 2]], 'int32'))->transpose();
$viewResult = CudaArray::minimum($view, new CudaArray([[3, 4], [5, 0]], 'int32'));
check_minimum($viewResult->toArray() === [[3, 4], [1, 0]], 'strided view');

$nan = new CudaArray([NAN, INF, -INF], 'float64');
$special = CudaArray::minimum($nan, new CudaArray([1.0, -INF, INF], 'float64'))->toArray();
check_minimum(is_nan($special[0]), 'NaN propagation');
check_minimum($special[1] === -INF && $special[2] === -INF, 'infinity ordering');

foreach (['float32', 'float64'] as $zeroDtype) {
    $positive = new CudaArray([0.0], $zeroDtype);
    $negative = new CudaArray([-0.0], $zeroDtype);
    check_minimum(has_zero_sign(CudaArray::minimum($positive, $positive), $zeroDtype, 0, false), "$zeroDtype +0,+0");
    check_minimum(has_zero_sign(CudaArray::minimum($negative, $negative), $zeroDtype, 0, true), "$zeroDtype -0,-0");
    check_minimum(has_zero_sign(CudaArray::minimum($positive, $negative), $zeroDtype, 0, true), "$zeroDtype +0,-0");
    check_minimum(has_zero_sign(CudaArray::minimum($negative, $positive), $zeroDtype, 0, true), "$zeroDtype -0,+0");
}

$zeroBroadcastTensor = CudaArray::minimum(
    new CudaArray([[0.0], [-0.0]], 'float64'),
    new CudaArray([[0.0, -0.0]], 'float64')
);
check_minimum(has_zero_sign($zeroBroadcastTensor, 'float64', 0, false), 'broadcast +0,+0');
check_minimum(has_zero_sign($zeroBroadcastTensor, 'float64', 1, true), 'broadcast +0,-0');
check_minimum(has_zero_sign($zeroBroadcastTensor, 'float64', 2, true), 'broadcast -0,+0');
check_minimum(has_zero_sign($zeroBroadcastTensor, 'float64', 3, true), 'broadcast -0,-0');

$empty = CudaArray::minimum(CudaArray::zeros([0, 3], 'float32'), CudaArray::zeros([1, 3], 'float32'));
check_minimum($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty broadcast');

$independent = CudaArray::minimum($a, $b);
$a[0][0] = 99;
check_minimum($independent->toArray()[0][0] === 1, 'independent result');

foreach ([
    [new CudaArray([1, 2], 'int32'), new CudaArray([1, 2, 3], 'int32')],
] as [$left, $right]) {
    try {
        CudaArray::minimum($left, $right);
        throw new RuntimeException('invalid minimum accepted');
    } catch (Cuda\InvalidArgumentException $error) {
    }
}

echo "minimum checked\n";
?>
--EXPECT--
minimum checked
