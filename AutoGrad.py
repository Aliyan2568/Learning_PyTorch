import torch
import numpy as np

a = torch.tensor([2.,3.], requires_grad=True)
b = torch.tensor([6.,4.], requires_grad=True)

f = 3 * a**3 - b**2

print(f)

f.backward(gradient=torch.tensor([1,1])) #Designates importance
print(a.grad) #Derivative value calculation
print(9 * a**2)