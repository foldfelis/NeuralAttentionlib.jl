using MLDatasets: CIFAR10
using Flux
using CUDA
using JLD2

MODEL_PATH = joinpath(@__DIR__, "../model/")

function load_data(mode=:train)
    train_x, train_y = mode === :train ? CIFAR10.traindata() : CIFAR10.testdata()

    return Float32.(train_x), Flux.onehotbatch(train_y, 0:9)
end

function save_model(m, model_name::String)
    jldsave(joinpath(MODEL_PATH, "$model_name.jld2"); model=cpu(m))
end

function get_model(model_name::String)
    f = jldopen(joinpath(MODEL_PATH, "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end

function vgg16(a)
    return Chain(
        Conv((3, 3), 3=>64, pad=2, a),
        BatchNorm(64),
        Conv((3, 3), 64=>64, pad=2, a),
        BatchNorm(64),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 64=>128, pad=2, a),
        BatchNorm(128),
        Conv((3, 3), 128=>128, pad=2, a),
        BatchNorm(128),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 128=>256, pad=2, a),
        BatchNorm(256),
        Conv((3, 3), 256=>256, pad=2, a),
        BatchNorm(256),
        Conv((3, 3), 256=>256, pad=2, a),
        BatchNorm(256),
        MaxPool((2, 2), stride=(2, 2)),

        Conv((3, 3), 256=>512, pad=2, a),
        BatchNorm(512),
        Conv((3, 3), 512=>512, pad=2, a),
        BatchNorm(512),
        Conv((3, 3), 512=>512, pad=2, a),
        BatchNorm(512),
        MaxPool((2, 2), stride=(2, 2)),

        flatten,

        Dense(7*7*512, 4096, a),
        Dropout(0.5),
        Dense(4096, 4096, a),
        Dropout(0.5),
        Dense(4096, 10),
        softmax,
    )
end

function train(model; batchsize=128, η₀=5e-4)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    train_loader = Flux.DataLoader(load_data(:train), batchsize=batchsize, shuffle=true)
    test_loader = Flux.DataLoader(load_data(:test), batchsize=batchsize, shuffle=false)
    train_data = [(𝐱, 𝐲) for (𝐱, 𝐲) in train_loader] |> device

    m = model(gelu) |> device
    loss(𝐱, y) = Flux.crossentropy(m(𝐱), y)

    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in test_loader)/length(test_loader)
        @info "loss: $validation_loss"

        push!(losses, validation_loss)
        if validation_loss == minimum(losses)
            save_model(m, string(model))
            @warn "'$(string(model))' updated!"
        end
    end

    opt = Flux.Descent(η₀)
    for e in 1:50
        @time begin
            @info "epoch $e\n η = $(opt.eta)"
            Flux.train!(loss, params(m), train_data, opt)
            # (e ≥ 20) && (e%20 == 0) && (opt.eta /= 2)
            validate()
        end
    end

    return m
end
