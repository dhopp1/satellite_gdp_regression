module ImageProcessing

using
Images,
Statistics

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
    processed_image = load(img_path) .|>
        Gray |> x->
        imresize(x, dim_x, dim_y) |> x->
        channelview(x) |>
        mean_normalize_image
    return processed_image
end

function process_batch(key, target, dim_x, dim_y)
    img_nums = []
    img_data = reshape([],dim_x,dim_y,0)
    for image in readdir("../data/images")
        push!(img_nums, parse(Int64, replace(image, ".png" => "")))
        img_data = cat(img_data, reshape(ImageProcessing.process_image("../data/images/$(image)", dim_x, dim_y), (dim_x, dim_y, 1)); dims=3)
        println(image)
    end
    X = hcat(reshape(img_data, dim_x * dim_y, :))
    y = [key[findall(x->x==img_nums[i], key.id)[1], target][1] for i in 1:length(img_nums)]
    return X, y
end

end
