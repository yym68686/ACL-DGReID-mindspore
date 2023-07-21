import mindspore
q = mindspore.Parameter([])
print(len(q))
if len(q) == 0:
    q = mindspore.Parameter([0, 1, 2])
for i in range(2):
    x = 1
    if i == q[1]:
        print("q[0]",q[0])
        pass
# q = sorted([100, 1])
print(q)