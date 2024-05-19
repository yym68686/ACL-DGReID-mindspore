bn_ms2pt = {
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var"
}

# 互换键和值
pt_ms2bn = {value: key for key, value in bn_ms2pt.items()}

print(pt_ms2bn)

a = "layer4.2.bn1.beta"
print(a[12:])
def replace_substring(input_string):
    # 定义替换规则
    bn_ms2pt = {
        "gamma": "weight",
        "beta": "bias",
        "moving_mean": "running_mean",
        "moving_variance": "running_var"
    }

    # 分割字符串以定位最后一个部分
    parts = input_string.rsplit('.', 1)  # 从右边开始分割一次

    # 替换最后一个部分如果存在于字典中
    if parts[-1] in bn_ms2pt:
        parts[-1] = bn_ms2pt[parts[-1]]

    # 重新组合字符串
    return '.'.join(parts)

# 示例字符串
input_string = "layer1.2.bn1.moving_mean"
# 进行替换
output_string = replace_substring(input_string)

print(output_string)