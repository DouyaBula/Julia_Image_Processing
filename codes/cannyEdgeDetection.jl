# 将浮点梯度矩阵映射为RGB图像
function map2RGB(img::Array{Float64, 2})
    # 将img映射到uint8中, 以便写入文件
    img = img./maximum(img) .* 255
    img = round.(UInt8, img)
    img_rgb = zeros(UInt8, size(img, 1), size(img, 2), 3)
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb[:, :, 3] = img
    return img_rgb
end

function map2RGB(img::Array{UInt8, 2})
    img_rgb = zeros(UInt8, size(img, 1), size(img, 2), 3)
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb[:, :, 3] = img
    return img_rgb
end




clear()
using TyImages
# Canny边缘检测

# 读取图像
img = imread("image1.png")

# 转换为灰度图
img_gray = rgb2gray(img)

# 1. 高斯滤波
# 1.1. 定义高斯滤波器
gaussian_filter = fspecial("gaussian", 5, 1.2)

# 1.2. 对图像进行高斯滤波
img_gaussian = imfilter(img_gray, gaussian_filter)

imshow(img_gaussian)

# 2. 使用Sobel算子计算梯度
# 2.1. 定义Sobel算子
sobel_x = fspecial("sobel")'
sobel_y = fspecial("sobel")

# 2.2. 计算梯度
img_sobel_x = imfilter(img_gaussian, sobel_x, method = "conv")
img_sobel_y = imfilter(img_gaussian, sobel_y, method = "conv")

# 2.3. 计算梯度幅值
img_sobel = sqrt.(img_sobel_x .^ 2 + img_sobel_y .^ 2)

imshow(img_sobel)

# 3. 非极大值抑制
# +-----+-----+
# |3 \ 4|1 / 2|
# +-----+-----+
# |2 / 1|4 \ 3|
# +-----+-----+

# 3.1. 计算梯度方向正切值
img_tan = img_sobel_y ./ img_sobel_x

# 3.2. 非极大值抑制
img_nms = zeros(size(img_sobel))
for i in 2:size(img_sobel, 1) - 1
    for j in 2:size(img_sobel, 2) - 1
        # 梯度值
        g = img_sobel[i, j]
        # 梯度方向在范围4
        if img_tan[i, j] >= 1
            r_val = img_sobel[i + 1, j] * (1 - 1/ img_tan[i, j]) + img_sobel[i + 1, j + 1] * (1 / img_tan[i, j])
            l_val = img_sobel[i - 1, j] * (1 - 1 / img_tan[i, j]) + img_sobel[i - 1, j - 1] * (1 / img_tan[i, j])
            if g > r_val && g > l_val
                img_nms[i, j] = g
            else
                img_nms[i, j] = 0
            end
        # 梯度方向在范围3
        elseif 0 <= img_tan[i, j] < 1
            r_val = img_sobel[i, j + 1] * (1 - img_tan[i, j]) + img_sobel[i + 1, j + 1] * img_tan[i, j]
            l_val = img_sobel[i, j - 1] * (1 - img_tan[i, j]) + img_sobel[i - 1, j - 1] * img_tan[i, j]
            if g > r_val && g > l_val
                img_nms[i, j] = g
            else
                img_nms[i, j] = 0
            end
        # 梯度方向在范围2
        elseif -1 <= img_tan[i, j] < 0
            img_tan[i, j] = -img_tan[i, j]
            r_val = img_sobel[i, j + 1] * (1 - img_tan[i, j]) + img_sobel[i - 1, j + 1] * img_tan[i, j]
            l_val = img_sobel[i, j - 1] * (1 - img_tan[i, j]) + img_sobel[i + 1, j - 1] * img_tan[i, j]
            if g > r_val && g > l_val
                img_nms[i, j] = g
            else
                img_nms[i, j] = 0
            end
        # 梯度方向在范围1
        elseif img_tan[i, j] < -1
            img_tan[i, j] = -img_tan[i, j]
            r_val = img_sobel[i - 1, j] * (1 - 1 / img_tan[i, j]) + img_sobel[i - 1, j + 1] * (1 / img_tan[i, j])
            l_val = img_sobel[i + 1, j] * (1 - 1 / img_tan[i, j]) + img_sobel[i + 1, j - 1] * (1 / img_tan[i, j])
            if g > r_val && g > l_val
                img_nms[i, j] = g
            else
                img_nms[i, j] = 0
            end
        end
    end
end

imshow(img_nms)

# 4. 双阈值检测
# 4.1. 定义高低阈值
max_val = maximum(img_nms)
min_val = minimum(img_nms)
high_threshold = max_val - (max_val - min_val) * 0.8
low_threshold = min_val + (max_val - min_val) * 0.1

# 4.2. 双阈值检测
img_double_threshold = zeros(UInt8, size(img_nms))
for i in 1:size(img_nms, 1)
    for j in 1:size(img_nms, 2)
        if img_nms[i, j] >= high_threshold
            img_double_threshold[i, j] = 255
        elseif img_nms[i, j] >= low_threshold
            img_double_threshold[i, j] = 128
        else
            img_double_threshold[i, j] = 0
        end
    end
end

imshow(img_double_threshold)

# 5. 边缘连接
# 5.1. 定义连接方向
directions = [
    1 0
    1 1
    0 1
    -1 1
]

# 5.2. 边缘连接
img_edge = deepcopy(img_double_threshold)
for i in 2:size(img_edge, 1) - 1
    for j in 2:size(img_edge, 2) - 1
        flag = false
        if img_edge[i, j] == 128
            for k in 1:size(directions, 1)
                if img_edge[i + directions[k, 1], j + directions[k, 2]] == 255
                    img_edge[i, j] = 255
                    flag = true
                    break
                end
            end
            if !flag
                img_edge[i, j] = 0
            end
        end
    end
end

# 6. 显示并写出结果
imshow(img_edge)
img_final = map2RGB(img_edge)
imwrite(img_final, "output_canny.png")