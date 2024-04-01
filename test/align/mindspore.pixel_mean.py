import mindspore
a = [123.675, 116.28, 103.53]
pixel_mean = mindspore.Tensor(a)
print(type(pixel_mean), pixel_mean)
# pixel_mean = pixel_mean.resize(1, 3, 1, 1)
pixel_mean = mindspore.ops.reshape(pixel_mean, (1, 3, 1, 1))

print(type(pixel_mean), pixel_mean)