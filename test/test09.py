import torch
import numpy as np
a = torch.arange(12,dtype=torch.float32).reshape(3,4)
print(a)
b = torch.tensor([[0,1,0,5,6,8],[1,2,0,0,5,0],[1,1,5,0,0,5]],dtype=torch.float32)
print(b)
print(b.nonzero())
print(b.zero_())
print(b)


# x = np.arange(12).reshape([2,2,3])
# print(x)
# a = np.copy(x[:,:,0])
# print(a)
# x[:,:,0]=x[:,:,2]
# # print(x)
# print(a)
# # print(x[:,:,2])
# x[:,:,2] = a
# print(x)