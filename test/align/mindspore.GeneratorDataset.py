import numpy as np
class MyDataset():
    """Self Defined dataset."""
    def __init__(self, n):
        self.data = []
        self.label = []
        for _ in range(n):
            self.data.append(np.zeros((3, 4, 5)))
            self.label.append(np.ones((1)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

import mindspore.dataset as ds

my_dataset = MyDataset(10)
# print(dir(my_dataset))
dataset = ds.GeneratorDataset(my_dataset, column_names=["data", "label"])
# print(dataset[0])

dataset = dataset.batch(5, drop_remainder=True)

# print(dataset[0])
# iterator = iter(dataset)
# # iterator = dataset.create_dict_iterator()

# print(next(iterator))
# print(next(iterator))
# print(len(dataset))
for data_dict in dataset:
    print(data_dict)
    # for name in data_dict.keys():
    #     print(name, data_dict[name].shape)