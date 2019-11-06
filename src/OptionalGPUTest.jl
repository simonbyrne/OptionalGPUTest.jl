module OptionalGPUTest

using CUDAapi, GPUifyLoops

export DefaultArray, mul2!, gpu_allowscalar

if CUDAapi.has_cuda()
    using CuArrays, CUDAnative
    device(::CuArray) = CUDA()
    gpu_allowscalar(x) = CuArrays.allowscalar(x)
    const DefaultArray = CuArrays.CuArray
else
    gpu_allowscalar(x) = nothing
    const DefaultArray = Array
end

device(::AbstractArray) = CPU()


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
