#~ import torch
#~ x = torch.Tensor(5, 3)
#~ print(x)
#~ x = torch.rand(5, 3)

#~ print("x:")
#~ print(x.size())
#~ print(x)

#~ print("y:")
#~ y = torch.rand(5, 3)
#~ print(y.size())
#~ print(y)

#~ print("x+y:")
#~ print(x+y)

#~ a = torch.ones(5)
#~ b = a.numpy()

#~ a.add_(1)
#~ print(a)
#~ print(b)

#~ import numpy as np
#~ import torch
#~ a = np.ones(5)
#~ b = torch.from_numpy(a)
#~ print(a)
#~ print(b)

#~ import torch
#~ from torch.autograd import Variable
#~ x = Variable(torch.ones(2, 2), requires_grad=True)
#~ print(x)
#~ y = x + 2
#~ print(y)
#~ print(y.grad_fn)

#~ z = y*y*3
#~ out = z.mean()
#~ print(z, out)

#~ out.backward()
#~ print(x.grad)
#~ x.grad=None
#~ gradients = torch.ones(2, 2)
#~ y.backward(gradients)
#~ print(x.grad)

#~ import torch
#~ from torch.autograd import Variable
#~ import numpy as np

#~ x = torch.randn(3)
#~ x = Variable(x, requires_grad=True)

#~ y = x*2
#~ while y.data.norm()<1000:
    #~ y = y*2
#~ print(y)

#~ gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
#~ print(gradients)
#~ y.backward(gradients)
#~ print(x.grad)

#~ x = torch.randn(3, 3)
#~ x = Variable(x, requires_grad=True)

#~ y = x*2
