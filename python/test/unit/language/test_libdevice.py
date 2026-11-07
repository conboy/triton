import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [
        ("j0", "bessel_j0"),
        ("j1", "bessel_j1"),
        ("y0", "bessel_y0"),
        ("y1", "bessel_y1"),
        ("cyl_bessel_i0", "i0"),
        ("cyl_bessel_i1", "i1"),
    ],
)
def test_bessel(dtype_str, libdevice_fn, torch_special_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    torch.manual_seed(42)
    x = torch.randn((SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    kernel[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)


def test_clz(device):

    @triton.jit
    def triton_clz(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.clz(tl.load(in_ptr + offsets)))

    x = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_clz[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([31, 30, 30, 29, 29, 28, 28, 27], dtype=torch.int32))


def test_popc(device):

    @triton.jit
    def triton_popc(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.popc(tl.load(in_ptr + offsets)))

    x = torch.tensor([0, 1, 2, 3, 4, 7, 8, 15], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_popc[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([0, 1, 1, 2, 1, 3, 1, 4], dtype=torch.int32))


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_isinf(device, dtype_str):

    @triton.jit
    def triton_isinf(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        in_tile = tl.load(in_ptr + offsets, mask=mask)
        if in_ptr.dtype.element_ty == tl.float32:
            out_tile = libdevice.finitef(in_tile)
        else:
            out_tile = libdevice.isfinited(in_tile)
        tl.store(out_ptr + offsets, out_tile, mask=mask)

    x = torch.tensor(
        [float(1), -float(1),
         float(0), -float(0),
         float("inf"), -float("inf"),
         float("nan"), -float("nan")], device=device, dtype=getattr(torch, dtype_str))
    res = torch.tensor([True, True, True, True, False, False, False, False])
    numel = x.numel()
    y = torch.empty_like(x, dtype=torch.bool)
    BLOCK_SIZE = 256
    triton_isinf[(triton.cdiv(numel, BLOCK_SIZE), )](x, y, numel, BLOCK_SIZE)
    assert torch.equal(y.cpu(), res)


def _uniform(lo, hi):
    return lambda size, dtype, device: torch.empty(size, dtype=dtype, device=device).uniform_(lo, hi)


def _randn(size, dtype, device):
    return torch.randn(size, dtype=dtype, device=device)


def _randint(size, dtype, device):
    return torch.randint(-20, 20, (size, ), dtype=torch.int32, device=device)


# fn name -> (reference evaluated in fp64, one sampler per argument)
_MATH_CASES = {
    "cbrt": (lambda x: x.sign() * x.abs().pow(1 / 3), [_randn]),
    "rcbrt": (lambda x: 1 / (x.sign() * x.abs().pow(1 / 3)), [_randn]),
    "exp10": (lambda x: torch.pow(10.0, x), [_randn]),
    "erfcinv": (lambda x: torch.erfinv(1 - x), [_uniform(0.1, 1.9)]),
    "tgamma": (lambda x: torch.lgamma(x).exp(), [_uniform(0.5, 5.0)]),
    "logb": (lambda x: torch.frexp(x).exponent.to(x.dtype) - 1, [_randn]),
    "remainder": (lambda x, y: x - y * torch.round(x / y), [_randn, _uniform(0.5, 2.0)]),
    "rhypot": (lambda x, y: 1 / torch.hypot(x, y), [_randn, _randn]),
    "fdim": (lambda x, y: (x - y).clamp(min=0), [_randn, _randn]),
    "scalbn": (lambda x, n: torch.ldexp(x, n), [_randn, _randint]),
    "sinpi": (lambda x: torch.sin(torch.pi * x), [_randn]),
    "cospi": (lambda x: torch.cos(torch.pi * x), [_randn]),
    "normcdf": (torch.special.ndtr, [_randn]),
    "normcdfinv": (torch.special.ndtri, [_uniform(0.01, 0.99)]),
    "norm3d": (lambda x, y, z: (x * x + y * y + z * z).sqrt(), [_randn] * 3),
    "rnorm3d": (lambda x, y, z: 1 / (x * x + y * y + z * z).sqrt(), [_randn] * 3),
    "norm4d": (lambda x, y, z, w: (x * x + y * y + z * z + w * w).sqrt(), [_randn] * 4),
    "rnorm4d": (lambda x, y, z, w: 1 / (x * x + y * y + z * z + w * w).sqrt(), [_randn] * 4),
}


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize("libdevice_fn", sorted(_MATH_CASES))
def test_math(dtype_str, libdevice_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)
    ref_fn, samplers = _MATH_CASES[libdevice_fn]

    torch.manual_seed(42)
    args = [sample(SIZE, dtype, device) for sample in samplers]
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = ref_fn(*[a.double() if a.is_floating_point() else a for a in args]).to(dtype)

    @triton.jit
    def kernel(a_p, b_p, c_p, d_p, out_p, fn: tl.constexpr, NARGS: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        a = tl.load(a_p + off)
        if NARGS == 1:
            res = getattr(libdevice, fn)(a)
        elif NARGS == 2:
            res = getattr(libdevice, fn)(a, tl.load(b_p + off))
        elif NARGS == 3:
            res = getattr(libdevice, fn)(a, tl.load(b_p + off), tl.load(c_p + off))
        else:
            res = getattr(libdevice, fn)(a, tl.load(b_p + off), tl.load(c_p + off), tl.load(d_p + off))
        tl.store(out_p + off, res)

    ptrs = args + [args[0]] * (4 - len(args))
    kernel[(1, )](*ptrs, y_exp, fn=libdevice_fn, NARGS=len(args), SIZE=SIZE)

    tol = {"float32": 1e-5, "float64": 1e-12}[dtype_str]
    torch.testing.assert_close(y_exp, y_ref, rtol=tol, atol=tol)
