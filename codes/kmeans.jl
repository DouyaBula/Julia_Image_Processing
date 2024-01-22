# kmeans++ 初始点选取
function select_centroids(data, k::Int)
    centroids = zeros(UInt8, k, size(data, 2))
    # 先随机选取一个簇中心
    centroids[1, :] = data[rand(1:size(data, 1)), :]
    # 选取剩余的簇中心
    for i in 2:k
        # 计算每个点到最近簇中心的距离
        distances = zeros(size(data, 1))
        for j in 1:size(data, 1)
            # 转换为整型，否则会溢出
            distances[j] = sum((
                convert(Array{Int32,1}, data[j, :]) - convert(Array{Int32,1}, centroids[1, :])) .^ 2)
            for m in 2:i-1
                distances[j] = min(distances[j], sum((data[j, :] - centroids[m, :]) .^ 2))
            end
        end

        # 选取距离最大的点作为新的簇中心
        centroids[i, :] = data[argmax(distances), :]
    end

    # 输出选取的簇中心, 按十进制
    println("Original centroids:")
    for i in 1:size(centroids, 1)
        println(join(centroids[i, :], ", "))
    end
    # 返回簇中心
    return centroids
end


# 聚类
function kmeans(data, k::Int, max_iters::Int)
    # 初始化簇中心
    centroids = select_centroids(data, k)

    # 初始化聚类标签
    labels = Vector{Int8}(undef, size(data, 1))

    for i in 1:max_iters
        # 输出进度
        print("Running: $i / $max_iters")

        # 分配每个点到最近的簇    
        for j in 1:size(data, 1)
            # 转换为整型，否则会溢出
            label = argmin([sum(
                (convert(Array{Int32,1}, data[j, :]) - convert(Array{Int32,1}, centroids[m, :]))
                .^
                2) for m in 1:k])
            labels[j] = label
        end

        # 更新簇中心为每个簇内点的平均值
        for j in 1:k
            if (!isempty(findall(labels .== j)))
                centroids[j, :] = round.(UInt8, mean(data[findall(labels .== j), :], dims=1))
            end
        end

        # 输出进度, 使用ANSI控制符清除当前行
        if i != max_iters
            print("\u001b[2K\r")
        else
            print("\n")
        end
    end

    return labels, centroids
end

function resize_image(image_array, max_size::Int)
    # 如果图像太大, 则降采样
    ratio = size(image_array, 1) / max_size
    image_array_resized = zeros(UInt8, max_size, round(Int, size(image_array, 2) / ratio), size(image_array, 3))
    if ratio > 1
        # 手动降采样, 防止出现浮点数
        for i in 1:300
            for j in 1:round(Int, size(image_array, 2) / ratio)
                for k in 1:size(image_array, 3)
                    image_array_resized[i, j, k] =
                        image_array[floor(Int, i * ratio), floor(Int, j * ratio), k]
                end
            end
        end
    end
    return ratio > 1 ? image_array_resized : image_array
end

clear()

using TyImages

# 读取图像
image_array = imread("image4.png")  # 替换图像文件路径

# 如果图像太大, 则降采样
image_array = resize_image(image_array, 300)

# 转回UInt8
image_array = round.(UInt8, image_array)

# 将图像数组重新组织成一维数组（以便于聚类）
reshaped_image = reshape(image_array, :, size(image_array, 3))

# 设置要分割的区域数目（簇的数量）和最大迭代次数
num_clusters = 4
max_iterations = 10

# 执行 K-means 聚类
labels, centroids = kmeans(reshaped_image, num_clusters, max_iterations)

# 输出选取的簇中心, 按十进制
println("Final centroids:")
for i in 1:size(centroids, 1)
    println(join(centroids[i, :], ", "))
end


# 将聚类标签重新组织成图像形状
segmented_image = reshape(labels, size(image_array, 1), size(image_array, 2))

# 将灰度图像映射为RGB图像
# 1. 生成几个颜色映射
color_map = [
    255 0 0
    0 255 0
    0 0 255
    255 255 0
    255 0 255
    0 255 255
    128 0 0
    0 128 0
    0 0 128
    128 128 0
]
# 2. 新建RGB图像
segmented_image_rgb = zeros(UInt8, size(image_array, 1), size(image_array, 2), 3)
# 3. 将每个簇的像素映射为对应的颜色
for i in 1:size(segmented_image, 1)
    for j in 1:size(segmented_image, 2)
        segmented_image_rgb[i, j, :] = color_map[segmented_image[i, j], :]
    end
end

# 显示并写出结果
imshow(segmented_image_rgb)
imwrite(segmented_image_rgb, "output_kmeans.png")