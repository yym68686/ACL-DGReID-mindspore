import mindspore

a = mindspore.Tensor(12, dtype=mindspore.float32)
output = mindspore.ops.arange(a)
# output = mindspore.ops.arange(a)
print(output)

print(output.dtype)

b = mindspore.Tensor(1)
print(b, type(b), int(b))

