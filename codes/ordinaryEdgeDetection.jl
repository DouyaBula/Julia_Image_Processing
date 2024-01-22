# 将浮点梯度矩阵映射为RGB图像
function map2RGB(img::Array{Float32,2})
    # 将img映射到uint8中, 以便写入文件
    img = img ./ maximum(img) .* 255
    img = round.(UInt8, img)
    img_rgb = zeros(UInt8, size(img, 1), size(img, 2), 3)
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb[:, :, 3] = img
    return img_rgb
end

function map2RGB(img::Array{UInt8,2})
    img_rgb = zeros(UInt8, size(img, 1), size(img, 2), 3)
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb[:, :, 3] = img
    return img_rgb
end

# 定义各种算子
# 输入一个字符串, 返回一个算子
function get_operator(operator::String)
    if operator == "roberts"
        return [1 0; 0 -1], [0 1; -1 0]
    elseif operator == "prewitt"
        return [-1 0 1; -1 0 1; -1 0 1], [-1 -1 -1; 0 0 0; 1 1 1]
    elseif operator == "sobel"
        return [-1 0 1; -2 0 2; -1 0 1], [-1 -2 -1; 0 0 0; 1 2 1]
    elseif operator == "laplacian4"
        return [0 1 0; 1 -4 1; 0 1 0], [0 1 0; 1 -4 1; 0 1 0]
    elseif operator == "laplacian8"
        return [1 1 1; 1 -8 1; 1 1 1], [1 1 1; 1 -8 1; 1 1 1]
    else
        # error
        println("Error: operator not found")
        exit()
    end
end




clear()
using TyImages
# 普通边缘检测

# 读取图像
img = imread("image1.png")

# 转换为灰度图
img_gray = rgb2gray(img)

# 用所有算子各自检测一遍
for operator in ["roberts", "prewitt", "sobel", "laplacian4", "laplacian8"]
    img_x = zeros(Float32, size(img_gray, 1), size(img_gray, 2))
    img_y = zeros(Float32, size(img_gray, 1), size(img_gray, 2))
    img_grad = zeros(Float32, size(img_gray, 1), size(img_gray, 2))
    # 获取算子
    operator_x, operator_y = get_operator(operator)
    # 检测
    img_x = imfilter(img_gray, operator_x)
    if operator_x != operator_y
        img_y = imfilter(img_gray, operator_y)
    end
    # 计算梯度
    if operator_x != operator_y
        img_grad = sqrt.(img_x .^ 2 .+ img_y .^ 2)
    else
        img_grad = abs.(img_x)
    end
    # 映射为RGB图像
    img_rgb = map2RGB(img_grad)
    # 写入文件
    imwrite(img_rgb, "output_$(operator).png")
end