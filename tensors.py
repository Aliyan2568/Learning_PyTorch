import torch
print(torch.cuda.is_available()) #Will return false since I don't have nvidia on my laptop

### Tensors (A high dimensional array)

import numpy as np

arr = np.array([[1,2,3],[4,5,6]])

tensor = torch.tensor([[1,2,3],[4,5,6]])
#tensor acts similar to numpy features that make computations easier such as sum or multiplying values in an array

print(arr.sum)
print(tensor.sum)
print(arr*5)
print(tensor*5)

#We can create tensors from numpy

tensor = torch.from_numpy(arr)
print(tensor)

print(arr.device)
