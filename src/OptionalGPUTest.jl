module OptionalGPUTest

using CuArrays, GPUifyLoops, CUDAnative

export DefaultArray, mul2!, gpu_allowscalar

__init__() = global DefaultArray = CuArrays.functional() ? CuArray : Array

gpu_allowscalar(x) = CuArrays.allowscalar(x)

device(::AbstractArray) = CPU()
device(::CuArray) = CUDA()

function mul2_kernel!(A)
    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

function mul2!(A)
    @launch(device(A), threads = length(A), mul2_kernel!(A))
    return A
end

end # module
