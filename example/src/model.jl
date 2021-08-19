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

function load_data(mode=:train)
    train_x, train_y = mode === :train ? CIFAR10.traindata() : CIFAR10.testdata()

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

train_loader = Flux.DataLoader(load_data(:train), batchsize=128, shuffle=true)
test_loader = Flux.DataLoader(load_data(:test), batchsize=128, shuffle=false)
train_data = [(𝐱, 𝐲) for (𝐱, 𝐲) in train_loader] |> device

function validate()
    validation_losses = [loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in test_loader]
    @info "loss: $(sum(validation_losses)/length(test_loader))"
end

call_back = Flux.throttle(validate, 60, leading=false, trailing=true)
Flux.@epochs 500 @time(Flux.train!(loss, params(m), train_data, Flux.ADAM(3f-4), cb=call_back))
