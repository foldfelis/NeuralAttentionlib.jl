using MLDatasets: CIFAR10
using Flux
using CUDA

if has_cuda()
    @info "CUDA is on"
    device = gpu
    CUDA.allowscalar(false)
else
    device = cpu
end

function load_data()
    train_x, train_y = CIFAR10.traindata()

    return Float32.(train_x) |> device, Flux.onehotbatch(train_y, 0:9) |> device
end

function vgg16()
    σ = relu

    return Chain(
        Conv((3, 3), 3=>64, pad=2, σ),
        Conv((3, 3), 64=>64, pad=2, σ),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 64=>128, pad=2, σ),
        Conv((3, 3), 128=>128, pad=2, σ),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 128=>256, pad=2, σ),
        Conv((3, 3), 256=>256, pad=2, σ),
        Conv((3, 3), 256=>256, pad=2, σ),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 256=>512, pad=2, σ),
        Conv((3, 3), 512=>512, pad=2, σ),
        Conv((3, 3), 512=>512, pad=2, σ),
        MaxPool((2, 2), stride=(2, 2)),

        flatten,

        Dense(7*7*512, 4096, σ),
        Dense(4096, 4096, σ),
        Dense(4096, 10),
        softmax,
    ) |> device
end

m = vgg16()
loss(𝐱, y) = Flux.logitcrossentropy(m(𝐱), y)

train_loader = Flux.DataLoader(load_data(), batchsize=128, shuffle=true)
data = [(𝐱, 𝐲) for (𝐱, 𝐲) in train_loader] |> device

Flux.@epochs 500 @time(Flux.train!(loss, params(m), data, Flux.ADAM(3f-4)))
