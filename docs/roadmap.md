### 1. Resumo executivo

A auditoria foi realizada somente por inspeção, reflexão PHP e execução de testes existentes. Nenhum arquivo foi alterado, nenhuma dependência foi instalada e nenhum build foi executado.

Conclusões principais:

- `php-gpu-tensors` é atualmente uma extensão essencialmente CUDA.
- A compilação depende de CUDA Toolkit, `nvcc`, headers CUDA e CUB.
- O tensor interno depende diretamente de estruturas CUDA, como `cudaMalloc`, `cudaMemcpy`, streams e device pointers.
- Não existe backend CPU equivalente implementado.
- `ZMatrix` possui uma superfície pública significativamente maior: 109 métodos públicos em `ZMatrix\ZTensor`.
- `ZMatrix` usa `float32`, armazenamento host com `std::vector<float>` e backend GPU opcional.
- O CPU é o caminho padrão do ZMatrix; `toGpu()` deve ser chamado explicitamente.
- O ZMatrix possui integração BLAS, OpenMP, SIMD e CUDA.
- O `php-gpu-tensors` possui uma arquitetura CUDA mais especializada, incluindo:
  - kernels CUDA;
  - cuBLAS/cuBLASLt;
  - operações assíncronas;
  - Fusion;
  - CUDA Graphs;
  - views;
  - múltiplos dtypes;
  - importação/exportação de buffers.
- O `php-gpu-tensors` não possui autograd identificado na API pública.
- O autograd do ZMatrix é experimental e incompleto.
- A primeira etapa recomendada é preparar uma compilação CPU-only mínima, sem adicionar novas operações públicas.

O workspace não está limpo. Existem alterações e arquivos não rastreados anteriores à auditoria nos dois projetos. Portanto, as conclusões representam o estado atual dos working trees, não necessariamente o último commit oficial.

---

### 2. Arquitetura do php-gpu-tensors

#### 2.1 Organização

Principais componentes encontrados:

```text
src/
├── cuda.c
├── cuda_wrapper.cpp
├── cuda_array/
│   ├── cuda_array_ce.c
│   ├── ca_private.c
│   ├── tensor_factory.c
│   ├── tensor_transfer.c
│   ├── tensor_import.c
│   ├── npy_import.c
│   └── tensor_where.c
├── cuda/
│   ├── cuda_kernels.cu
│   ├── matmul_kernels.cu
│   ├── reduction_ops.cu
│   ├── activation_kernels.cu
│   ├── broadcast_ops.cu
│   ├── scalar_ops.cu
│   ├── concat_kernels.cu
│   ├── memory_pool.c
│   └── nn_backend.cu
├── fusion.c
├── fusion_plan.c
├── fusion_execute.c
├── fusion_codegen.c
├── fusion_cache.c
├── fusion_capture.c
├── compiler_ce.c
├── kernel_ce.c
├── number_ce.c
├── nn.c
├── tensor.c
├── data_types.c
└── module_ce.c
```

Arquivos relevantes:

- `config.m4`
- `src/tensor.h`
- `src/cuda_array/ca_struct.h`
- `src/cuda_array/cuda_array_ce.c`
- `src/cuda_array/ca_arginfo.h`
- `src/fusion.c`
- `src/fusion_plan.c`
- `src/fusion_execute.c`
- `src/fusion_codegen.c`
- `src/cuda/matmul_kernels.cu`
- `src/cuda/memory_pool.c`
- `tests/`

#### 2.2 Linguagens e compilação

Linguagens utilizadas:

- C para integração com Zend/PHP;
- C++ para wrappers e integração de bibliotecas;
- CUDA C++ para kernels;
- PHP para testes, exemplos e scripts;
- shell para build e execução dos testes.

O arquivo `config.m4` contém:

```m4
PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda             Include CUDA support])
```

O caminho CUDA é obrigatório quando a extensão é configurada com suporte CUDA. O processo verifica:

- `cuda_runtime.h`;
- `nvcc`;
- `libnvrtc`;
- `libcuda`;
- `libcudart`;
- CUB;
- opcionalmente cuBLAS;
- opcionalmente cuBLASLt;
- opcionalmente cuDNN.

O build atual falha sem CUDA Toolkit. O próprio `config.m4` termina com erro quando não encontra:

```text
CUDA headers
nvcc
CUB headers
CUDA runtime libraries
```

Isso confirma que executar operações na CPU e compilar sem CUDA são requisitos ainda não atendidos pela arquitetura atual.

#### 2.3 API PHP exportada

As classes públicas detectadas foram:

```text
Cuda\Exception
Cuda\RuntimeException
Cuda\InvalidArgumentException
Cuda\OutOfMemoryException
Cuda\CompilationException
Cuda\Number
Cuda\Compiler
Cuda\CompiledModule
Cuda\ContiguousArray
Cuda\Fusion
Cuda\FusionGraph
Cuda\FusionExecution
Cuda\NN
Cuda\CudaArray
```

Funções globais exportadas:

```text
cuda_get_backend_info()
cuda_get_device_count()
cuda_get_device_info()
cuda_set_device()
cuda_get_current_device()
cuda_get_memory_info()
cuda_device_reset()
cuda_synchronize()
cuda_get_driver_version()
cuda_get_runtime_version()
cuda_get_last_error()
cuda_clear_error()
cuda_get_peer_access()
```

A classe principal de tensores é:

```text
Cuda\CudaArray
```

Métodos principais:

```text
__construct()
fromBuffer()
fromFlatArray()
fromFile()
fromNpy()
where()
slice()
concat()
multiply()
divide()
add()
subtract()
matmul()
transpose()
power()
sqrt()
exp()
log()
sin()
cos()
tan()
abs()
neg()
floor()
ceil()
round()
gt()
lt()
eq()
ne()
ge()
le()
sum()
mean()
max()
min()
prod()
argMax()
argMin()
getShape()
getStrides()
getNdims()
getSize()
toArray()
toHost()
toBuffer()
reshape()
flatten()
zeros()
ones()
full()
rand()
astype()
dtype()
```

#### 2.4 Representação do tensor

Em `src/tensor.h`, o tensor é representado por `tensor_t`:

```c
typedef struct tensor
{
    void *data;
    dtype_t dtype;
    int *shape;
    size_t element_size;
    size_t *strides;
    int ndims;
    size_t total_size;
    int ref_count;
    size_t allocated_size;
    int is_view;
    size_t offset;
    slice_info_t *slices;
    struct tensor *base_tensor;
    int num_slices;
    size_t *d_strides;
    int *d_shape;
    int is_on_gpu;
    int host_pinned;
    int is_dirty;
    int is_contiguous_cached;
    struct fusion_node *fusion;
    int fusion_failed;
    size_t fusion_readers;
} tensor_t;
```

Características confirmadas:

- `data` representa o buffer principal;
- `shape` representa dimensões;
- `strides` representam acesso multidimensional;
- `dtype` representa o tipo numérico;
- `is_view` diferencia views de buffers próprios;
- `base_tensor` mantém relação com o tensor original;
- `offset` representa deslocamento;
- `d_shape` e `d_strides` são metadados no device;
- `is_on_gpu` registra residência;
- `fusion` permite representação preguiçosa de operações;
- `ref_count` está presente na estrutura.

O `cuda_array_obj`, em `src/cuda_array/ca_struct.h`, mantém:

```c
zend_object std;
tensor_t *tensor_handle;
zend_array *shape;
```

#### 2.5 Gerenciamento de memória

As funções principais são:

```text
cuda_tensor_create_with_dtype()
cuda_tensor_create_empty_with_dtype()
cuda_tensor_destroy()
cuda_tensor_create_view()
cuda_tensor_create_sliced_view()
cuda_tensor_create_dim_view()
tensor_cast()
```

Há também:

- pool de memória em `src/cuda/memory_pool.c`;
- buffers pinned;
- buffers device;
- views compartilhadas;
- transferência host/device;
- controle de contiguidade;
- suporte a buffers assíncronos em algumas rotas.

O ciclo de vida é acoplado ao lifetime dos objetos Zend por meio dos object handlers da extensão.

Riscos relevantes:

- views dependem do lifetime do `base_tensor`;
- operações assíncronas mantêm referências internas;
- a introdução de buffers CPU exigirá separar claramente owner, view e residência;
- `void *data` não define sozinho se o buffer é host ou device.

#### 2.6 Dtypes

`src/data_types.h` declara suporte para:

```text
float32
float64
int8
int16
int32
int64
uint8
uint16
uint32
uint64
bool
list
```

Há funções para:

```text
dtype_from_string()
dtype_from_zval()
promote_types()
promote_types_for_comparison()
promote_types_for_logical()
promote_types_for_arithmetic()
tensor_cast()
can_safely_cast_to()
can_cast_unsafe()
```

O suporte declarado é mais amplo que o suporte necessariamente implementado por todos os kernels. A equivalência com ZMatrix é limitada porque o ZMatrix usa `float32` como representação principal.

#### 2.7 Broadcasting

O `php-gpu-tensors` contém:

```text
src/cuda/broadcast_ops.cu
src/cuda/broadcast_ops.cuh
src/cuda/broadcast_ops.h
```

Também existem funções de broadcast em:

```text
fusion_broadcast_cast
cuda_tensor_where
operações binárias
```

O suporte existe, mas deve ser analisado operação por operação. Não há evidência de que todos os métodos aceitem a mesma regra geral de broadcasting.

No ZMatrix, o broadcasting observado em `add()` e `sub()` é mais restrito:

- shapes iguais;
- matriz `[M,N]` com vetor `[N]`;
- algumas rotas de broadcast reverso;
- escalar;
- tensor de shape `[1]` tratado como escalar.

Isso não equivale ao broadcasting geral do NumPy.

#### 2.8 Execução CUDA

O projeto possui três camadas:

1. wrappers PHP/Zend;
2. representação interna de operações;
3. execução de kernels CUDA ou bibliotecas NVIDIA.

Arquivos importantes:

```text
src/cuda/cuda_kernels.cu
src/cuda/matmul_kernels.cu
src/cuda/reduction_ops.cu
src/cuda/activation_kernels.cu
src/cuda/broadcast_ops.cu
src/cuda/scalar_ops.cu
src/cuda/concat_kernels.cu
src/cuda/nn_backend.cu
```

Também existem:

- streams;
- operações assíncronas;
- sincronização explícita;
- cache de launch configurations;
- cache de módulos JIT;
- carregamento de PTX;
- NVRTC;
- compilação de kernels definidos em PHP.

#### 2.9 cuBLAS, cuBLASLt e cuDNN

O `config.m4` verifica:

```text
cublasCreate_v2
cublasLtMatmul
```

O projeto pode compilar com:

```text
HAVE_CUBLAS
HAVE_CUBLASLT
HAVE_CUDNN
```

As bibliotecas não são equivalentes a uma garantia de que todas as operações utilizam essas APIs.

O backend contém caminhos próprios para:

- matmul;
- reduções;
- operações elementwise;
- ativações;
- convolução;
- pooling.

A disponibilidade de cuDNN é opcional. Quando não está disponível, operações de rede neural podem ser consideradas indisponíveis ou encaminhadas para caminhos alternativos.

#### 2.10 Fusion

Fusion está implementado em:

```text
src/fusion.c
src/fusion_plan.c
src/fusion_execute.c
src/fusion_codegen.c
src/fusion_capture.c
src/fusion_cache.c
src/fusion_internal.h
```

A API pública inclui:

```text
Cuda\Fusion::run()
Cuda\Fusion::compile()
Cuda\Fusion::getCacheStats()
Cuda\Fusion::clearCache()

Cuda\FusionGraph::run()
Cuda\FusionGraph::runAsync()
Cuda\FusionGraph::getStats()
Cuda\FusionGraph::setProfiling()
Cuda\FusionGraph::getPlan()
Cuda\FusionGraph::getSource()

Cuda\FusionExecution::wait()
Cuda\FusionExecution::isFinished()
```

O fluxo identificado é:

```text
operações PHP
    ↓
fusion nodes
    ↓
fusion plan
    ↓
geração de CUDA source
    ↓
NVRTC/PTX
    ↓
módulo CUDA
    ↓
execução
```

#### 2.11 CUDA Graphs

Há suporte parcial/condicional a CUDA Graphs dentro dos planos de Fusion.

O código expõe informações como:

```text
cudaGraphCompatible
cudaGraphIncompatibility
cudaGraphRequested
```

A existência dessas propriedades não significa que todos os planos sejam compatíveis com CUDA Graphs.

#### 2.12 Autograd

Não foram encontradas classes ou métodos públicos de autograd no `php-gpu-tensors`.

Não há equivalente identificado para:

```text
requiresGrad()
backward()
getGrad()
zeroGrad()
addAutograd()
```

Portanto, autograd está ausente na API pública auditada.

---

### 3. Arquitetura do ZMatrix

#### 3.1 Organização

O núcleo está concentrado principalmente em:

```text
src/zmatrix.cpp
src/zmatrix_methods.h
src/zmatrix_arginfo.h
src/gpu_wrapper.h
src/gpu_kernels.cu
src/gpu_kernels.h
src/simd/
```

O arquivo `src/zmatrix.cpp` contém:

- estrutura `ZTensor`;
- armazenamento;
- shape e strides;
- operações CPU;
- gerenciamento GPU;
- integração BLAS;
- autograd;
- chamadas aos wrappers CUDA.

A camada Zend PHP é implementada principalmente em:

```text
src/zmatrix_methods.h
src/zmatrix_arginfo.h
```

#### 3.2 Representação interna

`ZTensor` possui:

```cpp
SharedTensorBuffer data;
std::vector<size_t> shape;
std::vector<size_t> strides;
size_t offset;
bool owns_data;
```

O armazenamento host é:

```cpp
std::vector<float>
```

O wrapper `SharedTensorBuffer` utiliza:

```cpp
std::shared_ptr<std::vector<float>>
```

Portanto, o ZMatrix é essencialmente `float32`.

Com CUDA habilitado, o tensor possui também:

```cpp
mutable float *d_data;
mutable size_t d_capacity;
mutable bool d_async_allocation;
mutable bool device_valid;
mutable bool host_valid;
mutable bool device_write_pending;
mutable std::mutex device_mutex;
```

A residência é controlada por duas cópias possíveis:

```text
host_valid
device_valid
```

#### 3.3 Política de residência

O comportamento documentado e confirmado no código é:

- criação normalmente no host;
- operações CPU são o comportamento padrão;
- `toGpu()` chama `ensure_device()`;
- `toCpu()` chama `ensure_host()` e invalida a residência device;
- várias operações GPU chamam `ensure_device()` implicitamente;
- algumas operações CUDA geram resultados diretamente no device;
- `toArray()` pode provocar sincronização device-to-host.

Isso significa que o ZMatrix tem um modelo CPU-first com promoção explícita ou implícita para GPU.

#### 3.4 Paralelismo CPU

Foram encontrados:

- OpenMP;
- intrinsics AVX/AVX2/AVX512;
- loops serializados para cargas menores;
- `std::thread`;
- `std::future`;
- `std::mutex`.

Existe uma constante:

```cpp
#define ZMATRIX_PARALLEL_THRESHOLD 10000000
```

O próprio código indica que o OpenMP é evitado para algumas cargas menores porque pode ser mais lento.

#### 3.5 BLAS

O `config.m4` procura:

```text
OpenBLAS
CBLAS
```

A configuração tenta localizar:

```text
cblas_dgemm
```

O código usa explicitamente:

```cpp
cblas_sgemm()
```

A integração BLAS está confirmada para pelo menos:

- `matmul`;
- `conv2d`, via transformação para GEMM.

Não foi encontrada evidência de que todas as operações elementwise utilizem BLAS.

#### 3.6 CUDA

O backend CUDA é condicional a:

```cpp
#ifdef HAVE_CUDA
```

A integração está em:

```text
src/gpu_kernels.cu
src/gpu_wrapper.h
```

O wrapper declara kernels para:

- add;
- sub;
- mul;
- div;
- ReLU;
- LeakyReLU;
- sigmoid;
- tanh;
- exp;
- abs;
- operações escalares;
- matmul;
- Adam;
- pow;
- log;
- sqrt;
- clip;
- softmax;
- broadcast;
- tile;
- cumsum;
- dot;
- matvec;
- transpose;
- reduções;
- argmax/argmin.

Contudo, a presença de uma função no wrapper não garante que todos os métodos públicos estejam sempre usando GPU. Muitas operações públicas chamam `ensure_host()` e executam no CPU.

#### 3.7 Autograd

O ZMatrix contém:

```cpp
AutogradNode
requires_grad
grad
grad_fn
parents
backward_fn
```

Métodos públicos:

```text
requiresGrad()
isRequiresGrad()
ensureGrad()
zeroGrad()
getGrad()
backward()
addAutograd()
subAutograd()
mulAutograd()
sumAutograd()
```

O próprio README registra que o autograd é experimental.

Limitações confirmadas:

- backward exige tensor escalar com shape `{1}`;
- operações inplace podem invalidar contexto;
- propagação é incompleta;
- o comportamento de `requires_grad` com mutações pode ser inconsistente;
- a infraestrutura existe, mas não há cobertura completa de todas as operações.

---

### 4. Inventário completo das operações ZMatrix

A superfície foi confirmada por:

1. reflexão da extensão instalada;
2. tabela `zmatrix_ztensor_methods`;
3. `src/zmatrix_methods.h`;
4. `src/zmatrix_arginfo.h`.

Foram identificados 109 métodos públicos próprios de `ZMatrix\ZTensor`.

#### 4.1 Construção e utilitários

| Método | Assinatura refletida |
|---|---|
| `__construct` | `($dataOrShape = optional)` |
| `arr` | `($input_data)` |
| `zeros` | `(array $shape)` |
| `ones` | `(array $shape)` |
| `full` | `(array $shape, ?float $value = optional)` |
| `identity` | `(int $size)` |
| `random` | `(array $shape, ?float $min = optional, ?float $max = optional)` |
| `randomUniform` | `(array $shape, float $minimum, float $maximum, int $seed)` |
| `randn` | `(array $shape, float $mean = optional, float $std_dev = optional)` |
| `arange` | `(float $arg1, ?float $arg2 = optional, float $arg3 = optional)` |
| `linspace` | `(float $start, float $stop, int $num = optional, bool $endpoint = optional)` |
| `logspace` | `(float $start, float $stop, int $num = optional, bool $endpoint = optional, float $base = optional)` |
| `eye` | `(int $N, ?int $M = optional, int $k = optional)` |
| `fill` | `(float $value)` |
| `copy` | `()` |
| `safe` | `($input, ?bool $copy = optional)` |
| `tile` | `(ZTensor $tensor, int $times)` |
| `concat` | `(array $tensors, int $axis = optional)` |
| `stack` | `(array $tensors)` |

Observações:

- o tipo numérico real é `float32`;
- shapes são vetores de dimensões positivas ou zero dependendo do caminho;
- o construtor também aceita outro `ZTensor`;
- vários métodos de criação são estáticos;
- `fill()` é mutável e retorna o próprio tensor.

#### 4.2 Operações aritméticas

| Método | Assinatura |
|---|---|
| `add` | `($other)` |
| `sub` | `($other)` |
| `mul` | `($other)` |
| `divide` | `($other)` |
| `scalarMultiply` | `(float $scalar)` |
| `scalarDivide` | `($scalar)` |
| `pow` | `(float $exponent)` |
| `minimum` | `($a, $b)` |
| `maximum` | `($a, $b)` |
| `adamUpdate` | `(ZTensor $gradient, ZTensor $firstMoment, ZTensor $secondMoment, float $learningRate, float $beta1, float $beta2, float $epsilon, float $biasCorrection1, float $biasCorrection2)` |

`add`, `sub` e operações semelhantes podem aceitar:

- escalares;
- arrays;
- outros tensores;
- broadcasting 2D×1D em casos específicos.

Importante: `add`, `sub`, `mul` e operações relacionadas são mutáveis em algumas rotas, retornando o próprio objeto PHP.

#### 4.3 Álgebra linear

| Método | Assinatura |
|---|---|
| `matmul` | `($other)` |
| `dot` | `($other)` |
| `transpose` | `()` |
| `permute` | `(array $axes)` |
| `reshape` | `(array $shape)` |
| `flatten` | `(int $startAxis = optional, int $endAxis = optional)` |
| `broadcast` | `(ZTensor $bias)` |
| `broadcastTo` | `(array $shape)` |
| `column` | `(int $col_idx)` |
| `row` | `(int $row_idx)` |
| `key` | `(array $indices)` |

`matmul()` utiliza CBLAS em CPU e caminhos CUDA quando o tensor está ou é transferido para GPU.

#### 4.4 Funções matemáticas e ativações

| Método |
|---|
| `abs()` |
| `sigmoid()` |
| `sigmoidDerivative()` |
| `relu()` |
| `reluDerivative()` |
| `tanh()` |
| `tanhDerivative()` |
| `leakyRelu(?float $alpha)` |
| `leakyReluDerivative(?float $alpha)` |
| `exp()` |
| `log()` |
| `sqrt()` |
| `softmax()` |
| `softmaxDerivative()` |

As funções existem na camada PHP e possuem implementação CPU. Algumas têm kernels CUDA.

#### 4.5 Reduções e estatística

| Método | Assinatura |
|---|---|
| `sum` | `($axis = optional, bool $keepDims = optional)` |
| `sumtotal` | `()` |
| `mean` | `($axis = optional, bool $keepDims = optional)` |
| `min` | `()` |
| `max` | `()` |
| `std` | `()` |
| `variance` | `($axis = optional, bool $keepDims = optional, int $correction = optional)` |
| `median` | `()` |
| `percentile` | `(float $q)` |
| `histogram` | `(int $bins = optional)` |
| `mode` | `(?int $axis = optional)` |
| `argmax` | `(?int $axis = optional)` |
| `argmin` | `(?int $axis = optional)` |
| `uniqueCounts` | `()` |
| `bincount` | `($weights = optional)` |
| `cumsum` | `(?int $axis = optional)` |

As reduções são predominantemente CPU no código do ZMatrix, ainda que existam algumas rotas CUDA específicas.

#### 4.6 Manipulação de shape e indexação

| Método |
|---|
| `shape()` |
| `ndim()` |
| `size()` |
| `isEmpty()` |
| `reshape()` |
| `permute()` |
| `flatten()` |
| `broadcastTo()` |
| `slice()` |
| `column()` |
| `row()` |
| `gather()` |
| `key()` |
| `argsort()` |
| `sort()` |

A assinatura refletida para `slice()` é genérica:

```text
slice($axis, $start, $end)
```

Views e materializações são tratadas internamente por `shape`, `strides`, `offset` e `owns_data`.

#### 4.7 Comparações e seleção

| Método |
|---|
| `greater()` |
| `where()` |
| `findIndicesWhere()` |
| `isin()` |
| `minimum()` |
| `maximum()` |
| `clip()` |

A semântica de `where()` do ZMatrix é específica:

```text
where(int $feature_index, float $threshold)
```

Não é o mesmo contrato de:

```text
CudaArray::where(condition, x, y)
```

#### 4.8 Ordenação e conjuntos

| Método |
|---|
| `unique()` |
| `uniqueCounts()` |
| `bincount()` |
| `argsort()` |
| `sort()` |
| `isin()` |
| `cumsum()` |
| `histogram()` |
| `median()` |
| `percentile()` |
| `mode()` |

#### 4.9 CNN

| Método | Assinatura |
|---|---|
| `im2col` | `(int $kernelHeight, int $kernelWidth, int $strideHeight = optional, int $strideWidth = optional, int $paddingHeight = optional, int $paddingWidth = optional)` |
| `col2im` | `(array $inputShape, int $kernelHeight, int $kernelWidth, int $strideHeight = optional, int $strideWidth = optional, int $paddingHeight = optional, int $paddingWidth = optional)` |
| `conv2d` | `(ZTensor $filters, ?ZTensor $bias = optional, int $strideHeight = optional, int $strideWidth = optional, int $paddingHeight = optional, int $paddingWidth = optional)` |
| `conv2dBackward` | `(ZTensor $gradOutput, ZTensor $filters, int $strideHeight = optional, int $strideWidth = optional, int $paddingHeight = optional, int $paddingWidth = optional)` |
| `maxPool2d` | `(int $kernelHeight, int $kernelWidth, int $strideHeight, int $strideWidth, int $paddingHeight = optional, int $paddingWidth = optional)` |
| `maxPool2dBackward` | `(ZTensor $gradOutput, ZTensor $indices, array $inputShape)` |
| `globalAveragePool2d` | `()` |
| `globalAveragePool2dBackward` | `($gradOutput, array $inputShape)` |

O contrato esperado para convolução é NCHW. O código valida shapes de quatro dimensões.

#### 4.10 Residência e gerenciamento de dispositivo

| Método |
|---|
| `toGpu()` |
| `toCpu()` |
| `isOnGpu()` |
| `freeDevice()` |

Esses métodos não são operações matemáticas, mas fazem parte da API pública e precisam ser considerados na equivalência.

#### 4.11 Autograd

| Método |
|---|
| `requiresGrad()` |
| `isRequiresGrad()` |
| `ensureGrad()` |
| `zeroGrad()` |
| `getGrad()` |
| `backward()` |
| `addAutograd()` |
| `subAutograd()` |
| `mulAutograd()` |
| `sumAutograd()` |

---

### 5. Matriz de equivalência

Classificação utilizada:

- `COMPLETA`: equivalente funcional confirmado para o mesmo caso;
- `PARCIAL`: existe operação relacionada, mas shape, dtype, mutabilidade ou parâmetros diferem;
- `AUSENTE`: não existe equivalente público;
- `INCOMPATÍVEL`: existe algo com nome ou objetivo semelhante, mas sem semântica equivalente;
- `NÃO VERIFICADA`: não foi possível confirmar completamente.

#### 5.1 Construção

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `__construct` | `CudaArray::__construct` | Sim | Não | Parcial | PARCIAL | Ambos aceitam arrays, mas os dtypes e a residência diferem. |
| `arr` | `CudaArray::__construct` | Sim | Não | Parcial | PARCIAL | `arr` aceita tensor ou array; CUDA constrói tensor device. |
| `zeros` | `CudaArray::zeros` | Sim | Não | Parcial | PARCIAL | CUDA aceita dtype; ZMatrix é float32. |
| `ones` | `CudaArray::ones` | Sim | Não | Parcial | PARCIAL | Mesmo caso, dtype diferente. |
| `full` | `CudaArray::full` | Sim | Não | Parcial | PARCIAL | Equivalente para shapes e valores simples. |
| `identity` | Nenhum | Não | Não | — | AUSENTE | Não há construtor público específico. |
| `random` | `CudaArray::rand` | Sim | Não | Parcial | PARCIAL | Intervalo e defaults precisam ser comparados. |
| `randomUniform` | `CudaArray::rand` | Sim | Não | Parcial | PARCIAL | ZMatrix exige seed explícita. |
| `randn` | Nenhum | Não | Não | — | AUSENTE | Não há gerador normal público. |
| `arange` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `linspace` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `logspace` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `eye` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `fill` | `full` | Sim | Não | Parcial | PARCIAL | `fill` é mutável; `full` cria novo tensor. |
| `copy` | Clone/construção indireta | Sim | Não | Parcial | PARCIAL | Não há `copy()` público equivalente confirmado. |
| `safe` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `tile` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `concat` | `CudaArray::concat` | Sim | Não | Parcial | PARCIAL | Eixo e requisitos de shape precisam de testes equivalentes. |
| `stack` | Nenhum | Não | Não | — | AUSENTE | Ausente. |

#### 5.2 Aritmética

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `add` | `add` | Sim | Não | Parcial | PARCIAL | ZMatrix pode mutar o próprio tensor e aceita broadcasting restrito. |
| `sub` | `subtract` | Sim | Não | Parcial | PARCIAL | Nome e mutabilidade diferem. |
| `mul` | `multiply` | Sim | Não | Parcial | PARCIAL | ZMatrix aceita array/escalar em mais rotas. |
| `divide` | `divide` | Sim | Não | Parcial | PARCIAL | Dtypes e tratamento de divisão por zero precisam de equivalência. |
| `scalarMultiply` | `multiply` | Sim | Não | Parcial | PARCIAL | CUDA aceita `other`, mas o contrato escalar não foi isolado. |
| `scalarDivide` | `divide` | Sim | Não | Parcial | PARCIAL | Sem método escalar específico. |
| `pow` | `power` | Sim | Não | Parcial | PARCIAL | Assinaturas diferentes. |
| `minimum` | Nenhum equivalente elementwise confirmado | Não | Não | Incompatível | INCOMPATÍVEL | `CudaArray::min()` é redução, não mínimo entre dois valores. |
| `maximum` | Nenhum equivalente elementwise confirmado | Não | Não | Incompatível | INCOMPATÍVEL | Mesmo problema de semântica. |
| `adamUpdate` | `Cuda\NN` não equivalente | Não | Não | Incompatível | INCOMPATÍVEL | Não há Adam público equivalente. |

#### 5.3 Matemática e ativações

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `abs` | `abs` | Sim | Não | Parcial | PARCIAL | Resultado e mutabilidade precisam de testes. |
| `sigmoid` | Nenhum método público | Não | Não | — | AUSENTE | Pode existir em kernels internos, mas não na API `CudaArray`. |
| `sigmoidDerivative` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `relu` | Nenhum método `CudaArray` | Indireto | Não | Parcial | PARCIAL | Existe backend NN/kernel, mas não método público equivalente. |
| `reluDerivative` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `tanh` | Nenhum método público | Indireto | Não | — | AUSENTE | Kernel interno não constitui API equivalente. |
| `tanhDerivative` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `leakyRelu` | Nenhum método público | Indireto | Não | — | AUSENTE | Wrapper CUDA existe, API pública não. |
| `leakyReluDerivative` | Nenhum | Indireto | Não | — | AUSENTE | Ausente. |
| `exp` | `exp` | Sim | Não | Parcial | PARCIAL | Sem backend CPU. |
| `log` | `log` | Sim | Não | Parcial | PARCIAL | Sem backend CPU. |
| `sqrt` | `sqrt` | Sim | Não | Parcial | PARCIAL | Sem backend CPU. |
| `softmax` | `Cuda\NN::softmax` | Sim | Não | Parcial | PARCIAL | API e eixos diferem. |
| `softmaxDerivative` | Nenhum | Não | Não | — | AUSENTE | Ausente. |

#### 5.4 Álgebra linear

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `matmul` | `CudaArray::matmul` | Sim | Não | Parcial | PARCIAL | Equivalente para matrizes compatíveis; shapes gerais precisam de testes. |
| `dot` | `matmul` | Sim | Não | Parcial | PARCIAL | Dot vetorial não é necessariamente equivalente a matmul. |
| `transpose` | `transpose` | Sim | Não | Parcial | PARCIAL | ZMatrix não recebe eixos; CUDA recebe eixos opcionais. |
| `permute` | `transpose` | Sim | Não | Parcial | PARCIAL | `permute` geral não possui equivalente confirmado. |
| `reshape` | `reshape` | Sim | Não | Parcial | PARCIAL | Semântica de view/cópia pode divergir. |
| `flatten` | `flatten` | Sim | Não | Parcial | PARCIAL | ZMatrix permite faixa de eixos; CUDA não. |
| `broadcast` | Broadcasting interno | Sim | Não | Parcial | PARCIAL | Método público CUDA não equivalente confirmado. |
| `broadcastTo` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `column` | Slicing/indexação | Sim | Não | Parcial | PARCIAL | Não há método `column`. |
| `row` | Slicing/indexação | Sim | Não | Parcial | PARCIAL | Não há método `row`. |
| `key` | `slice`/`__invoke` | Sim | Não | Parcial | PARCIAL | Semântica de indexação diferente. |

#### 5.5 Reduções e estatística

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `sum` | `sum` | Sim | Não | Parcial | PARCIAL | CUDA aceita eixo inteiro; ZMatrix aceita eixo flexível e `keepDims`. |
| `sumtotal` | `sum` sem eixo | Sim | Não | Parcial | PARCIAL | Resultado e shape precisam ser comparados. |
| `mean` | `mean` | Sim | Não | Parcial | PARCIAL | Mesmo problema de eixo. |
| `min` | `min` | Sim | Não | Parcial | PARCIAL | Ambos são redução, mas shape/eixo divergem. |
| `max` | `max` | Sim | Não | Parcial | PARCIAL | Mesmo problema. |
| `std` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `variance` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `median` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `percentile` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `histogram` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `mode` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `argmax` | `argMax` | Sim | Não | Parcial | PARCIAL | Diferença de capitalização e shape/eixo. |
| `argmin` | `argMin` | Sim | Não | Parcial | PARCIAL | Mesmo problema. |
| `uniqueCounts` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `bincount` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `cumsum` | Nenhum método público | Indireto | Não | — | AUSENTE | Kernel interno CUDA existe, mas API pública não. |

#### 5.6 Shape, indexação e seleção

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `shape` | `getShape` | Sim | Não | Parcial | PARCIAL | Nome diferente; shape equivalente. |
| `ndim` | `getNdims` | Sim | Não | COMPLETA | COMPLETA | Mesmo conceito. |
| `size` | `getSize` | Sim | Não | COMPLETA | COMPLETA | Mesmo conceito. |
| `isEmpty` | `getSize() === 0` | Sim | Não | Parcial | PARCIAL | Não há método direto. |
| `slice` | `slice`/`__invoke` | Sim | Não | Parcial | PARCIAL | Sintaxes diferentes. |
| `gather` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `argsort` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `sort` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `where` | `CudaArray::where` | Sim | Não | Incompatível | INCOMPATÍVEL | ZMatrix usa feature/threshold; CUDA usa condition/x/y. |
| `findIndicesWhere` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `isin` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `concat` | `concat` | Sim | Não | Parcial | PARCIAL | Eixos e validações precisam de testes. |
| `stack` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `unique` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `bincount` | Nenhum | Não | Não | — | AUSENTE | Ausente. |

#### 5.7 Comparação e utilidades

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `greater` | `gt` | Sim | Não | Parcial | PARCIAL | Operação elementwise equivalente em casos simples. |
| `minimum` | Nenhum | Não | Não | INCOMPATÍVEL | INCOMPATÍVEL | Não confundir com redução `min`. |
| `maximum` | Nenhum | Não | Não | INCOMPATÍVEL | INCOMPATÍVEL | Não confundir com redução `max`. |
| `clip` | Nenhum método público | Indireto | Não | — | AUSENTE | Kernel interno existe em alguns caminhos. |
| `mode` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `__toString` | `toArray` | Sim | Não | Parcial | PARCIAL | String e array possuem semânticas diferentes. |

#### 5.8 CNN

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `im2col` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `col2im` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `conv2d` | `Cuda\NN::conv2d` | Sim | Não | Parcial | PARCIAL | CUDA possui dilation/groups; ZMatrix usa contrato NCHW mais restrito. |
| `conv2dBackward` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `maxPool2d` | `Cuda\NN::pool2d` | Sim | Não | Parcial | PARCIAL | Assinaturas e modos diferem. |
| `maxPool2dBackward` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `globalAveragePool2d` | Nenhum | Não | Não | — | AUSENTE | Ausente. |
| `globalAveragePool2dBackward` | Nenhum | Não | Não | — | AUSENTE | Ausente. |

#### 5.9 Dispositivo

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado | Observações |
|---|---|---:|---:|---|---|---|
| `toGpu` | Construção CUDA | Sim | Não | Parcial | PARCIAL | CUDA já cria o tensor no backend device. |
| `toCpu` | `toHost`/`toArray` | Sim | Não | Parcial | PARCIAL | `toCpu()` do ZMatrix também invalida a residência device. |
| `isOnGpu` | Nenhum método de tensor | Sim | Não | Parcial | PARCIAL | Há informação global de backend, não equivalente direto por tensor. |
| `freeDevice` | Nenhum | Sim | Não | — | AUSENTE | Ausente. |

#### 5.10 Autograd

| Operação ZMatrix | Equivalente php-gpu-tensors | GPU | CPU | Compatibilidade | Estado |
|---|---|---:|---:|---|---|
| `requiresGrad` | Nenhum | Não | Não | — | AUSENTE |
| `isRequiresGrad` | Nenhum | Não | Não | — | AUSENTE |
| `ensureGrad` | Nenhum | Não | Não | — | AUSENTE |
| `zeroGrad` | Nenhum | Não | Não | — | AUSENTE |
| `getGrad` | Nenhum | Não | Não | — | AUSENTE |
| `backward` | Nenhum | Não | Não | — | AUSENTE |
| `addAutograd` | Nenhum | Não | Não | — | AUSENTE |
| `subAutograd` | Nenhum | Não | Não | — | AUSENTE |
| `mulAutograd` | Nenhum | Não | Não | — | AUSENTE |
| `sumAutograd` | Nenhum | Não | Não | — | AUSENTE |

---

### 6. Diagnóstico do backend CPU

#### 6.1 O que já existe

No `php-gpu-tensors`:

- tipos numéricos abstratos;
- shape;
- strides;
- views;
- operações de conversão;
- metadados host/device;
- mecanismo de ownership;
- separação parcial entre wrappers e kernels;
- funções de broadcast;
- funções de redução;
- funções de transferência;
- operações Fusion;
- suporte a operações assíncronas.

Esses elementos podem servir de base, mas não constituem um backend CPU funcional.

#### 6.2 O que falta

Faltam, no mínimo:

- modo de compilação sem CUDA;
- remoção da dependência incondicional de `cuda_runtime.h`;
- representação host válida sem `cudaMalloc`;
- allocator CPU;
- dispatch por device;
- kernels ou loops CPU;
- transferência host/device condicional;
- tratamento de objetos quando não existe GPU;
- testes CPU;
- definição de política para dtypes CPU;
- definição de comportamento para Fusion sem CUDA;
- definição de comportamento para NN sem cuDNN;
- definição de comportamento para streams e operações assíncronas em CPU.

#### 6.3 Distinções importantes

##### Executar operações na CPU

Significa que a extensão está compilada com suporte a CUDA, mas uma operação específica usa o host.

O ZMatrix já faz isso.

##### Executar sem GPU disponível

Significa que a extensão foi compilada com suporte CUDA, mas o runtime não encontra uma GPU.

O `php-gpu-tensors` atual tende a falhar ou lançar erro porque depende do contexto CUDA.

##### Compilar sem CUDA Toolkit

Significa que nem headers, `nvcc`, bibliotecas CUDA ou CUB estão presentes.

O `php-gpu-tensors` atual não suporta essa situação. O `config.m4` exige esses componentes.

Esses três cenários são diferentes.

#### 6.4 Alternativa A — backend CPU independente

Arquitetura:

```text
CudaArray
    ├── backend CUDA
    └── backend CPU
```

Vantagens:

- menor acoplamento entre CUDA e CPU;
- possibilidade de compilar sem CUDA;
- testes CPU mais simples;
- facilidade de usar OpenBLAS;
- menor risco de misturar estados host/device.

Desvantagens:

- duplicação inicial de operações;
- necessidade de definir uma interface comum;
- maior superfície de manutenção;
- impacto maior nos arquivos de tensor e dispatch.

Avaliação:

| Critério | Avaliação |
|---|---|
| Complexidade | Média/alta |
| Compatibilidade CUDA | Boa |
| Compilação sem CUDA | Boa |
| Testes | Boa |
| Manutenção | Média |
| BLAS/OpenBLAS | Boa |
| Fusion | Requer decisão separada |
| Autograd | Pode ser implementado depois |

#### 6.5 Alternativa B — abstração de dispositivos

Arquitetura:

```text
Tensor
 ├── Storage CPU
 └── Storage CUDA

Dispatcher
 ├── CpuOps
 └── CudaOps
```

Vantagens:

- melhor arquitetura de longo prazo;
- suporte natural a múltiplos devices;
- possibilidade de manter uma API única;
- separação clara de memória e execução.

Desvantagens:

- grande alteração estrutural;
- risco de regressões;
- exige revisar views, strides e Fusion;
- pode alterar a ABI interna;
- custo alto antes da primeira operação CPU.

Avaliação:

| Critério | Avaliação |
|---|---|
| Complexidade | Alta |
| Compatibilidade CUDA | Boa se bem isolada |
| Compilação sem CUDA | Excelente |
| Testes | Complexos |
| Manutenção | Boa depois da estabilização |
| BLAS/OpenBLAS | Excelente |
| Fusion | Requer backend específico |
| Autograd | Boa base futura |

#### 6.6 Alternativa C — híbrida incremental

Arquitetura:

- manter `tensor_t`;
- introduzir flags de backend;
- encapsular apenas alocação, destruição e dispatch;
- implementar inicialmente uma única operação CPU;
- manter os demais métodos CUDA-only.

Vantagens:

- menor risco;
- menor alteração;
- permite validar o modelo;
- preserva a API atual;
- facilita comparação CPU/GPU.

Desvantagens:

- arquitetura temporariamente híbrida;
- risco de condicionais espalhados;
- pode gerar estados incompletos;
- exige disciplina para não criar uma segunda arquitetura acidental.

Avaliação:

| Critério | Avaliação |
|---|---|
| Complexidade | Baixa/média |
| Compatibilidade CUDA | Melhor |
| Compilação sem CUDA | Possível gradualmente |
| Testes | Controláveis |
| Manutenção | Aceitável se limitada |
| BLAS/OpenBLAS | Possível |
| Fusion | Mantido CUDA-only inicialmente |
| Autograd | Adiado |

#### 6.7 Recomendação

A alternativa recomendada é a C, com uma regra:

```text
primeiro separar o lifecycle/storage;
depois adicionar uma única operação CPU;
depois repetir operação por operação.
```

A primeira infraestrutura não deve tentar abstrair toda a execução, Fusion, autograd e CUDA Graphs de uma vez.

---

### 7. Estratégia arquitetural recomendada

Recomendação:

```text
Fase 1:
  tornar possível compilar sem CUDA, sem adicionar operações

Fase 2:
  introduzir backend CPU mínimo para uma única operação

Fase 3:
  adicionar uma operação pública por etapa

Fase 4:
  criar dispatch comum CPU/CUDA

Fase 5:
  expandir reduções e shape operations

Fase 6:
  tratar NN, Fusion e autograd separadamente
```

A separação mínima necessária deve envolver:

```text
Tensor metadata
    shape
    strides
    dtype
    size

Storage
    CPU buffer
    CUDA buffer

Execution
    CPU operation
    CUDA operation

Device state
    CPU
    CUDA
```

Não é recomendável começar alterando toda a API `CudaArray`. A API pode continuar existindo enquanto internamente o tensor passa a ter um backend selecionável.

Uma possível política incremental:

```text
CudaArray::zeros(...)
    backend padrão CUDA se disponível

CudaArray::zeros(..., device: "cpu")
    backend CPU futuro

CudaArray::toCpu()
CudaArray::toGpu()
    conversões explícitas futuras
```

Mas essa API não deve ser implementada antes de definir:

- ownership;
- sincronização;
- lifetime;
- comportamento de views;
- dtype;
- tratamento de erro.

---

### 8. Riscos e compatibilidade

#### 8.1 Riscos confirmados

##### Dependência estrutural de CUDA

`src/tensor.h` inclui:

```cpp
#include <cuda_runtime.h>
```

Isso impede compilação sem headers CUDA.

##### Estado device acoplado ao tensor

`tensor_t` contém diretamente:

```text
is_on_gpu
d_shape
d_strides
data
```

Não existe uma abstração de storage CPU/CUDA independente.

##### Views e ownership

A estrutura contém:

```text
is_view
base_tensor
offset
ref_count
```

Um backend CPU precisa preservar a mesma semântica de lifetime.

##### Operações assíncronas

O Fusion e os módulos compilados mantêm:

- streams;
- eventos;
- operações pendentes;
- referências a objetos;
- buffers temporários.

Uma implementação CPU não pode simplesmente remover esses campos sem definir comportamento equivalente.

##### Exceções nativas

O código usa exceções CUDA/C++ convertidas para exceções PHP. Uma nova camada CPU terá que impedir que exceções C++ atravessem a ABI Zend.

##### Mutabilidade

O ZMatrix possui operações mutáveis como `add`, `sub`, `fill` e `softmax`.

O `php-gpu-tensors` tende a retornar novos objetos em operações elementwise. Uma equivalência direta pode alterar o comportamento esperado de aplicações.

##### Dtypes

O `php-gpu-tensors` declara mais dtypes que o ZMatrix, mas nem todo kernel necessariamente suporta todos eles.

##### NTS/ZTS

O PHP utilizado na auditoria é:

```text
PHP 8.5.4 NTS
```

Não foi validada a extensão em ambiente ZTS.

##### CUDA Graphs

O Fusion contém compatibilidade condicional com CUDA Graphs. Um backend CPU não deve tentar simular essa funcionalidade inicialmente.

#### 8.2 Riscos potenciais

- race conditions entre host e device;
- double free de buffers;
- views apontando para storage já liberado;
- conversões implícitas inesperadas;
- diferenças de arredondamento entre CPU e GPU;
- diferenças de shape em reduções;
- divergência de broadcasting;
- degradação de desempenho por cópias host/device;
- chamadas CUDA em um build CPU-only;
- uso de tipos CUDA em headers públicos;
- alterações na ABI da extensão;
- comportamento diferente em PHP 8.4 e PHP 8.5;
- problemas com OpenMP e exceções;
- alterações involuntárias na semântica de operações existentes;
- perda de referência Zend durante operações assíncronas;
- tentativas de uso de Fusion sem backend CUDA.

#### 8.3 Medidas preventivas

Cada etapa futura deve:

- alterar uma única operação ou infraestrutura mínima;
- adicionar teste positivo;
- adicionar teste negativo;
- testar shape inválido;
- testar tensor vazio;
- testar dtype;
- testar lifetime;
- testar equivalência CPU/GPU quando aplicável;
- atualizar stub somente quando a API mudar;
- executar regressão da API existente;
- não misturar autograd, Fusion e backend CPU na mesma etapa.

---

### 9. Estado atual dos testes

#### 9.1 php-gpu-tensors

Testes encontrados:

```text
tests/*.phpt
tests/check_api_surface.php
tests/ca_*.phpt
tests/fusion_*.phpt
tests/cuda_*.phpt
tests/nn_*.phpt
tests/memory_pool_test.c
tests/npy_header_test.c
tests/where_shape_test.c
```

Também existe:

```text
run-tests.sh
```

O script exige build prévio em um diretório como:

```text
cuda_build-8.5
```

e executa:

- verificação da API;
- teste de memory pool;
- teste de header NPY;
- teste de shape do `where`;
- testes PHPT;
- validação opcional de GPU.

#### 9.2 Teste executado: API pública

Comando:

```bash
php -n tests/check_api_surface.php
```

Resultado:

```text
Frozen PHP API surface matches the baseline.
```

Status: aprovado.

#### 9.3 ZMatrix

Testes encontrados:

```text
tests/test.sh
tests/basic.phpt
tests/complete.phpt
tests/core_validation.php
tests/cpu_no_cuda.php
tests/cpu_lifecycle.php
tests/strided_views.php
tests/adam_update.php
tests/cuda/
```

#### 9.4 Teste executado: CPU sem CUDA

Comando:

```bash
php tests/cpu_no_cuda.php
```

Resultado:

```text
CPU-only CUDA rejection passed.
```

Status: aprovado.

#### 9.5 Teste executado: validação central

Comando:

```bash
php tests/core_validation.php
```

Resultado:

```text
PASS shape multiplication overflow
PASS negative shape rejected
PASS reshape overflow
PASS invalid clip bounds
PASS NaN clip bound rejected
PASS CPU remains default
PASS lifecycle stress
All core validation checks passed.
```

Status: aprovado.

#### 9.6 Teste abrangente ZMatrix

Foi iniciado:

```bash
bash tests/test.sh
```

O teste confirmou aprovação de várias áreas:

- construtor;
- string;
- shape;
- size;
- ndim;
- empty;
- toArray;
- copy;
- key;
- zeros;
- ones;
- full;
- fill;
- arange;
- linspace;
- logspace;
- identity;
- eye;
- arr;
- random;
- randn;
- add;
- sub;
- mul;
- divide;
- scalarMultiply;
- scalarDivide;
- matmul;
- dot;
- abs;
- exp;
- log;
- sqrt;
- pow;
- column;
- argsort;
- where;
- sigmoid;
- relu;
- tanh;
- leakyRelu;
- softmax;
- derivadas;
- reduções básicas;
- comparações;
- clip.

A execução ultrapassou o limite de observação usado para a auditoria antes de apresentar um resumo final completo. Portanto, não classifico o teste inteiro como aprovado; somente as verificações exibidas foram confirmadas.

#### 9.7 CI

Não foi identificada uma configuração clara de CI no workspace do `php-gpu-tensors`.

Há scripts e documentação de testes locais, mas não foi confirmada uma pipeline automatizada que valide:

- CPU-only;
- GPU;
- PHP NTS;
- PHP ZTS;
- múltiplas versões PHP;
- ausência do CUDA Toolkit;
- equivalência CPU/GPU.

#### 9.8 Testes de memória

Existem testes e artefatos relacionados a:

```text
memory_pool_test.c
test_memory_audit.php
valgrind_output.txt
valgrind_zmatrix_only.txt
```

Entretanto, os arquivos de resultados existentes no workspace são artefatos anteriores e não foram tratados como execução nova da auditoria.

---

### 10. Roadmap incremental

A ordem recomendada é:

1. Infraestrutura mínima para separar headers CUDA dos headers comuns.
2. Infraestrutura mínima de compilação CPU-only.
3. Estrutura de storage host.
4. `CudaArray::zeros()` em CPU.
5. `CudaArray::ones()` em CPU.
6. `CudaArray::full()` em CPU.
7. `CudaArray::add()` em CPU.
8. `CudaArray::subtract()` em CPU.
9. `CudaArray::multiply()` em CPU.
10. `CudaArray::divide()` em CPU.
11. `CudaArray::scalar operations` em CPU.
12. `CudaArray::reshape()` em CPU.
13. `CudaArray::flatten()` em CPU.
14. `CudaArray::transpose()` em CPU.
15. `CudaArray::sum()` em CPU.
16. `CudaArray::mean()` em CPU.
17. `CudaArray::min()` em CPU.
18. `CudaArray::max()` em CPU.
19. `CudaArray::prod()` em CPU.
20. `CudaArray::argMin()` em CPU.
21. `CudaArray::argMax()` em CPU.
22. `CudaArray::abs()` em CPU.
23. `CudaArray::sqrt()` em CPU.
24. `CudaArray::exp()` em CPU.
25. `CudaArray::log()` em CPU.
26. `CudaArray::power()` em CPU.
27. `CudaArray::sin()` em CPU.
28. `CudaArray::cos()` em CPU.
29. `CudaArray::tan()` em CPU.
30. `CudaArray::concat()` em CPU.
31. `CudaArray::slice()` em CPU.
32. `CudaArray::where()` em CPU.
33. `CudaArray::matmul()` em CPU.
34. Integração OpenBLAS para `matmul`.
35. Conversão CPU↔GPU.
36. `Cuda\NN::conv2d()` em CPU.
37. `Cuda\NN::pool2d()` em CPU.
38. Política de comportamento de Fusion em CPU.
39. Política de comportamento de operações assíncronas em CPU.
40. Investigação independente de autograd.
41. Suporte futuro a CUDA Graphs somente quando aplicável.

Cada item de operação deve ser uma etapa independente, com:

- implementação;
- teste positivo;
- teste negativo;
- teste de limites;
- teste de shape;
- teste de dtype;
- teste CPU/GPU;
- stub;
- documentação;
- regressão.

---

### 11. Primeira etapa recomendada

#### Objetivo

Preparar a menor infraestrutura possível para permitir que o código seja compilado sem CUDA Toolkit, sem adicionar ainda nenhuma nova operação CPU.

A primeira etapa não deve implementar `add`, `matmul`, `sum` ou qualquer outra operação matemática.

#### Arquivos potencialmente envolvidos

Principalmente:

```text
config.m4
src/tensor.h
src/data_types.h
src/cuda_exceptions.h
src/module_ce.c
```

Possivelmente, posteriormente:

```text
src/cuda_array/ca_struct.h
src/cuda_array/cuda_array_ce.c
src/cuda_array/ca_private.c
```

#### Dependências

- PHP development headers;
- compilador C/C++;
- autoconf/phpize;
- nenhuma dependência CUDA para o modo CPU-only.

O modo CUDA existente deverá continuar exigindo:

- CUDA Toolkit;
- nvcc;
- headers;
- bibliotecas NVIDIA.

#### Critérios de aceitação

1. O projeto deve compilar em modo CPU-only sem `cuda_runtime.h`.
2. O projeto deve continuar compilando em modo CUDA.
3. O modo CUDA não pode alterar a ABI PHP atual.
4. A extensão deve carregar sem GPU quando compilada CPU-only.
5. Nenhuma nova operação pública deve ser adicionada.
6. Os testes de API existentes devem continuar passando.
7. O build deve falhar claramente quando for solicitado CUDA sem dependências.
8. Nenhum código CUDA deve ser referenciado no caminho CPU-only.
9. O modo NTS atual deve permanecer funcional.
10. A etapa deve incluir testes de build, carregamento e ausência de chamadas CUDA.

#### Testes necessários

- build CPU-only;
- carregamento da extensão;
- `php -m`;
- `check_api_surface.php`;
- teste de execução sem GPU;
- teste de compilação CUDA existente;
- teste de que headers CUDA não são incluídos no modo CPU-only;
- teste de erro quando CUDA é solicitado sem Toolkit.

#### Motivo da prioridade

Sem essa etapa, qualquer operação CPU futura continuará dependente de:

```text
cuda_runtime.h
nvcc
CUDA libraries
CUB
```

Implementar uma operação CPU antes de resolver essa dependência produziria apenas um caminho CPU parcial dentro de uma extensão que ainda não pode ser compilada sem CUDA.

Essa é a menor alteração estrutural que desbloqueia o restante do roadmap.

---

### 12. Evidências e limitações

#### 12.1 php-gpu-tensors

Evidências principais:

```text
php-gpu-tensors/config.m4
php-gpu-tensors/src/tensor.h
php-gpu-tensors/src/data_types.h
php-gpu-tensors/src/cuda_array/ca_struct.h
php-gpu-tensors/src/cuda_array/cuda_array_ce.c
php-gpu-tensors/src/cuda_array/ca_arginfo.h
php-gpu-tensors/src/fusion.c
php-gpu-tensors/src/fusion_plan.c
php-gpu-tensors/src/fusion_execute.c
php-gpu-tensors/src/fusion_internal.h
php-gpu-tensors/src/cuda/matmul_kernels.cu
php-gpu-tensors/src/cuda/memory_pool.c
php-gpu-tensors/tests/
php-gpu-tensors/run-tests.sh
```

API confirmada por reflexão:

```text
Cuda\CudaArray
Cuda\Compiler
Cuda\CompiledModule
Cuda\ContiguousArray
Cuda\Fusion
Cuda\FusionGraph
Cuda\FusionExecution
Cuda\NN
Cuda\Number
```

Teste confirmado:

```text
Frozen PHP API surface matches the baseline.
```

#### 12.2 ZMatrix

Evidências principais:

```text
zmatrix/src/zmatrix.cpp
zmatrix/src/zmatrix_methods.h
zmatrix/src/zmatrix_arginfo.h
zmatrix/src/gpu_wrapper.h
zmatrix/src/gpu_kernels.cu
zmatrix/config.m4
zmatrix/tests/
zmatrix/stubs/
zmatrix/README.md
```

A tabela de métodos é registrada em:

```text
src/zmatrix.cpp
```

A implementação dos métodos PHP está em:

```text
src/zmatrix_methods.h
```

A estrutura `ZTensor` está em:

```text
src/zmatrix.cpp
```

O armazenamento principal é:

```cpp
std::vector<float>
```

A integração BLAS foi confirmada por:

```cpp
cblas_sgemm()
```

A integração CUDA foi confirmada por:

```text
#ifdef HAVE_CUDA
src/gpu_wrapper.h
src/gpu_kernels.cu
```

Reflexão confirmou 109 métodos públicos próprios em `ZMatrix\ZTensor`.

Testes confirmados:

```text
CPU-only CUDA rejection passed.
All core validation checks passed.
Frozen PHP API surface matches the baseline.
```

#### 12.3 Limitações da auditoria

Não foi possível confirmar completamente:

- cobertura de todos os caminhos CUDA para cada método ZMatrix;
- desempenho real de cada operação;
- equivalência numérica completa entre os dois projetos;
- comportamento em PHP ZTS;
- comportamento em múltiplas versões de PHP;
- comportamento sem GPU do `php-gpu-tensors`;
- build limpo a partir de clone novo;
- uso efetivo de cuBLASLt em todas as operações possíveis;
- cobertura final completa do script `zmatrix/tests/test.sh`;
- estabilidade de todas as rotas de autograd;
- compatibilidade de todos os dtypes em todos os kernels;
- equivalência completa de broadcasting entre as extensões.

Também não foram considerados benchmarks como prova universal de superioridade. Resultados de desempenho dependem de:

- tamanho dos tensores;
- GPU;
- driver;
- versão CUDA;
- BLAS;
- OpenMP;
- quantidade de threads;
- transferências host/device;
- estado de aquecimento;
- custo de criação dos objetos PHP.

A primeira tarefa recomendada permanece exclusivamente de infraestrutura: permitir uma compilação CPU-only mínima, sem implementar ainda nenhuma operação pública.