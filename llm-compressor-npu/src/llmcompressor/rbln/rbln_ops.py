from typing import Callable

import torch

from .rbln_envs import USE_CUSTOM_OPS

def replace(*original_func: Callable):
    def decorator(custom_func: Callable):
        if not USE_CUSTOM_OPS:
            return custom_func
        for func in original_func:
            if hasattr(func, '__objclass__') and func.__objclass__ is torch._C.TensorBase:
                setattr(torch.Tensor, func.__name__, custom_func)
            elif hasattr(func, '__module__'):
                module_path = func.__module__
                func_qualname = func.__qualname__
                if module_path == "torch":
                    module = __import__(module_path, fromlist=[''])
                    setattr(module, func.__name__, custom_func)
                elif module_path == "torch._C._linalg":
                    module = __import__("torch.linalg", fromlist=[''])
                    if "linalg_" in func_qualname:
                        func_qualname = func_qualname.split("linalg_")[-1]
                    setattr(module, func_qualname, custom_func)
                else:
                    raise NotImplementedError(f"Replacing module {module_path} not implemented yet.")
            else:
                raise NotImplementedError(f"Cannot determine how to replace {func}. It doesn't appear to be a module function or tensor method.")
        return custom_func

    return decorator


@replace(torch.sqrt, torch.Tensor.sqrt)
def torch_sqrt(x: torch.Tensor, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.sqrt.
    """
    return torch.tensor(1, device=x.device, dtype=x.dtype) / torch.rsqrt(x)


@replace(torch.amin, torch.Tensor.amin)
def torch_amin(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False, keepdims: bool = False, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.amin using torch.min.
    
    Args:
        x: Input tensor
        dim: Dimension(s) to reduce. Can be int, tuple of ints, or None (reduce all dims)
        keepdim: Whether to keep the reduced dimensions
        out: Optional output tensor to store the result
    
    Returns:
        Minimum values along the specified dimension(s)
    """
    keepdim = keepdim or keepdims
    if dim is None:
        result = torch.min(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    if not keepdim:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.min(result, dim=d, keepdim=keepdim).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.amax, torch.Tensor.amax)
def torch_amax(x: torch.Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False, keepdims: bool = False, *, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.amax using torch.max.
    
    Args:
        x: Input tensor
        dim: Dimension(s) to reduce. Can be int, tuple of ints, or None (reduce all dims)
        keepdim: Whether to keep the reduced dimensions
        out: Optional output tensor to store the result
    
    Returns:
        Maximum values along the specified dimension(s)
    """
    keepdim = keepdim or keepdims
    if dim is None:
        result = torch.max(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    if not keepdim:
        dims = tuple(sorted(dims, reverse=True))
    
    result = x
    for d in dims:
        result = torch.max(result, dim=d, keepdim=keepdim).values
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result


@replace(torch.round, torch.Tensor.round)
def torch_round(x: torch.Tensor, *, decimals: int = 0, out: torch.Tensor | None = None):
    """
    Custom implementation of torch.round.
    
    Implements "round half to even" (banker's rounding) behavior.
    
    Args:
        x: Input tensor
        decimals: Number of decimal places to round to (default: 0)
        out: Optional output tensor to store the result
    
    Returns:
        Rounded tensor with same dtype as input
    """
    if decimals != 0:
        scale_factor = 10.0 ** decimals
        scaled_x = x * scale_factor
    else:
        scaled_x = x

    floor_x = torch.floor(scaled_x)
    frac_part = scaled_x - floor_x
    is_even = floor_x == 2 * torch.floor(floor_x / 2)
    is_half = torch.abs(frac_part - 0.5) < 1e-7
    should_round_up = frac_part >= 0.5
    should_round_up = torch.where(is_half, ~is_even, should_round_up)
    result = torch.where(should_round_up, floor_x + 1, floor_x)

    if decimals != 0:
        result = result / scale_factor
    if out is not None:
        out.copy_(result)
        return out
    
    return result
