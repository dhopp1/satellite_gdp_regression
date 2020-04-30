include("ImageProcessing.jl")

using
CSV,
DataFrames,
Flux,
Statistics,
.ImageProcessing

# image key
key = CSV.read("../data/key.csv") |> DataFrame

# load images
reload = false

dim_x = 240
dim_y = 240
target = :population
n_epochs = 100

if reload
    X, y = ImageProcessing.process_batch(key, target, dim_x, dim_y)
    CSV.write("../data/X.csv", DataFrame(X))
    CSV.write("../data/y.csv", DataFrame(y=y))
else
    X = CSV.read("../data/X.csv") |> Array
    y = CSV.read("../data/y.csv") |> Array |> x-> reshape(x, size(y)[1])
end
model_data = Iterators.repeated((X, y), n_epochs)

# define model
model = Chain(
    Dense(dim_x * dim_y, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 1, relu)
)
loss(x, y) = Flux.mae(model(x), y)
opt = Flux.ADAM()
ps = Flux.params(model)
accuracy(x, y) = mean(abs.(model(x) .- y))
evalcb = Flux.throttle(() -> @show(accuracy(X, y)), 1)

# model training
Flux.train!(loss, ps, model_data, opt, cb = evalcb)

# inference
test_id = 10
test_path = "../data/images/$(test_id).png"
pred = ImageProcessing.process_image(test_path, dim_x, dim_y) |> x->
    reshape(x, dim_x * dim_y, :) |>
    model |> x->
    Float64(x[1])
actual = key[findall(x->x==test_id, key.id)[1], target][1]
abs(pred - actual)
