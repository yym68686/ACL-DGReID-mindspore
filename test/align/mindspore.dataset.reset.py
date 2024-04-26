import mindspore.dataset as ds
import numpy as np
import os
os.system("clear")
class MyIterable:
    def __init__(self):
        self._index = 0
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)

dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])
s = iter(dataset)
# s = dataset.create_dict_iterator(num_epochs=3)
while True:
    try:
        i = next(s)
    except StopIteration:
        s = iter(dataset)
        i = next(s)
    print(i)
dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
iterator = dataset.create_dict_iterator(num_epochs=10)
for item in iterator:
    # item is a dict
    print(item)