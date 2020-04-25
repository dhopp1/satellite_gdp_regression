using Colors, CSV, DataFrames, Flux, Images, Plots, Statistics

function mean_normalize_image(img)
    """pass channelview, returns floats"""
    img = Float64.(img)
    val_dict = Dict(
        1 => (mean(img[1,:,:]), std(img[1,:,:])),
        2 => (mean(img[2,:,:]), std(img[2,:,:])),
        3 => (mean(img[3,:,:]), std(img[3,:,:]))
    )
    for channel in 1:3
        img[channel,:,:] = (img[channel,:,:] .- val_dict[channel][1]) ./ val_dict[channel][2]
    end
    return img
end

function process_image(img_path, dim_x, dim_y)
    img = load(img_path)
    processed_image = img |> x->
        imresize(x, dim_x, dim_y) |> x->
        channelview(x)[1:3,:,:] |>
        mean_normalize_image
    return Dict("original_image" => img, "processed_image" => processed_image)
end

# image key
key = CSV.read("key.csv") |> DataFrame

# load images
dim_x = 240
dim_y = 240

img_nums = []
img_data = []
for image in readdir("images")
    push!(img_nums, parse(Int64, replace(image, ".png" => "")))
    push!(img_data, process_image("images/$(image)", dim_x, dim_y)["processed_image"])
    println(image)
end
