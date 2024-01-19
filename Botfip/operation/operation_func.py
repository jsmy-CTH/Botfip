import numpy as np
import sympy as sp
import scipy as sc
import torch
from torchquad import  Simpson
from torch.func import jacrev, vmap,grad,vjp

def sp_subtract(a, b):
    return sp.Add(a, sp.Mul(-1, b))

def sp_div(a, b):
    return a/b

def np_div(a, b):
    if b!=0:
        return a/b
    else:
        return a/1e-8

def np_relu(x):
    return np.maximum(np.zeros_like(x),x)

def torch_relu(x):
    return torch.maximum(torch.zeros_like(x),x)

def sp_relu(x):
    return sp.Max(0,x)




def torch_div(a, b):
    return a/b

def np_sqrt(x):
    return np.sqrt(np.abs(x))

def torch_sqrt(x):
    return torch.sqrt(torch.abs(x))

def sp_sqrt(x):
    return sp.sqrt(sp.Abs(x))

def np_log(x):
    if x==0:
        return 0
    else:
        return np.log(np.abs(x))

def torch_log(x):
    return torch.log(torch.abs(x))

def sp_log(x):
    return sp.log(sp.Abs(x))

def np_C(x,parameters=None):
    return parameters[0]

def sp_C(x,parameters=None):
    return parameters[0]

def torch_C(x,parameters=None):
    return parameters[0]

def np_integrate(x,func = None,index = 0):
    if isinstance(x,(tuple, list)):
        dim = len(x)
    elif isinstance(x,np.ndarray):
        dim = x.shape[-1]
    else:
        raise TypeError("x should be a list or a np.ndarray")
    assert index < dim,"index out of range"
    #print('func:',func)
    def partial_func(y):
        new_x = np.copy(x)
        new_x[index] = y
        return func(new_x)
    x_slice = np.copy(x)[index]
    return sc.integrate.quad(partial_func,0,x_slice)[0]

def torch_integrate(x, func=None, index=0):
    if isinstance(x, (tuple, list)):
        dim = len(x)
    elif isinstance(x, torch.Tensor):
        dim = x.shape[-1]
    else:
        raise TypeError("x should be a list or a torch.Tensor")
    assert index < dim, "index out of range"
    vec_func = torch.func.vmap(func, in_dims=0)

    def partial_func(y):
        new_x = x.clone()
        y_shape = y.shape
        y = y.reshape(-1)
        y_dim = y.shape[0]
        new_x = new_x.repeat(y_dim,1)
        new_x[:,index] = y
        return vec_func(new_x)

    simpson = Simpson()
    x_slice = x[index]
    integration_domain = [[0, x_slice]]
    result = simpson.integrate(partial_func, dim=1, N=101, integration_domain=integration_domain)
    return result


def sp_integrate(x,func = None,index = 0):
    if isinstance(x,(tuple, list)):
        dim = len(x)
        assert index < dim,"index out of range"
        return sp.integrate(func(x), x[index])
    elif isinstance(x,sp.Basic):
        return sp.integrate(func(x), x)
    else:
        raise TypeError("x should be a list or a sp.Basic")


def np_diff(x,func = None,index = 0):
    if isinstance(x, (tuple, list)):
        dim = len(x)
    elif isinstance(x,np.ndarray):
        dim = x.shape[-1]
    else:
        raise TypeError("x should be a list or a np.ndarray")
    assert index < dim,"index out of range"

    def partial_func(y):
        new_x = np.copy(x)
        new_x[index] = y
        return func(new_x)
    x_slice = np.copy(x)[index]
    return sc.misc.derivative(partial_func,x_slice,dx=1e-6)

def torch_diff(x,func = None,index = 0):
    if isinstance(x, (tuple, list)):
        dim = len(x)
    elif isinstance(x,torch.Tensor):
        dim = x.shape[-1]
    else:
        raise TypeError("x should be a list or a np.ndarray")
    assert index < dim,"index out of range"
    grad_f = jacrev(func)
    return grad_f(x)[index]

def sp_diff(x,func = None,index = 0):
    if isinstance(x,(tuple, list)):
        dim = len(x)
        assert index < dim,"index out of range"
        return sp.diff(func(x), x[index])
    elif isinstance(x,sp.Basic):
        return sp.diff(func(x), x)
    else:
        raise TypeError("x should be a list or a sp.Basic")

def np_linear(x,parameters=None):
    w = parameters[0]
    b = parameters[1]
    return w*x+b

def torch_linear(x,parameters=None):
    w = parameters[0]
    b = parameters[1]
    return w*x+b

def sp_linear(x,parameters=None):
    w = parameters[0]
    b = parameters[1]
    return sp.Add(sp.Mul(w,x),b)

def np_pow(x,parameters=None):
    w = parameters[0]
    if w<1:
        return np.abs(x)**w
    else:
        return x**w

def torch_pow(x,parameters=None):
    w = parameters[0]
    return torch.abs(x)**w

def sp_pow(x,parameters=None):
    w = parameters[0]
    return x ** w
    #if w<1 and (int(1/w)-1/w==0.0):
    #    return sp.root(x,int(1/w))
    #else:
def np_inv(x,parameters=None):
    w = parameters[0]
    return w/x

def sp_inv(x,parameters=None):
    w = parameters[0]
    return w/x

def torch_inv(x,parameters=None):
    w = parameters[0]
    return w/x

def sp_cbrt(x):
    return sp.root(x,3)

def torch_cbrt(x):
    return x**(1/3)


def np_slice(x,index = 0):
    return x[index]

def sp_slice(x,index = 0):
    return x[index]

def torch_slice(x,index = 0):
    return x[index]

def sp_square(x):
    return x**2

"""
def np_diff(x,func = None,index = 0):
    return sc.misc.derivative(func,x,dx=1e-6)

def sp_diff(x,func = None,index = 0):
    return sp.diff(func(x),x)

"""