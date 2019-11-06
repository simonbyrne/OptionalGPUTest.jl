# OptionalGPUTest.jl

Testing optional GPU support for Julia packages

```
using OptionalGPUTest
gpu_allowscalar(false)

A = DefaultArray(ones(100))
mul2!(A)
```