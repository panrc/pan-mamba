# Copyright (c) 2023, Tri Dao, Albert Gu.
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numbers

# Set functions to None if they cannot be imported, to force fallback to Python implementation
causal_conv1d_fn, causal_conv1d_update = None, None
selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None
selective_state_update = None
RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def _selective_scan_py(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Pure Python implementation of selective scan.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    D = D.float() if D is not None else None
    z = z.float() if z is not None else None
    
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdnl', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdnl', delta, B, u)

    for i in range(u.shape[2]):
        x = deltaA[..., i] * x + deltaB_u[..., i]
        y = torch.einsum('bdn,bn->bd', x, C[..., i])
        ys.append(y)
        
    y = torch.stack(ys, dim=2)
    
    if D is not None:
        y = y + u * D[..., None]
    
    if z is not None:
        y = y * F.silu(z)

    out = y.to(dtype=dtype_in)
    
    if return_last_state:
        return out, x
    else:
        return out

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = False # FORCE SLOW PATH
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2" or bimamba_type=='v3':
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            self.in_proj_extra = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, extra_emb=None):
        B, L, _ = hidden_states.shape
        
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1) # (B, L, D_in)

        x = x.permute(0, 2, 1) # (B, D_in, L)
        x = self.conv1d(x)[:, :, :L]
        x = self.act(x)
        x = x.permute(0, 2, 1) # (B, L, D_in)

        x_dbl = self.x_proj(x)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt).permute(0, 2, 1) # (B, D_in, L)
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        B_proj = B_proj.permute(0, 2, 1) # (B, N, L)
        C_proj = C_proj.permute(0, 2, 1) # (B, N, L)

        y = _selective_scan_py(x.permute(0,2,1), dt, A, B_proj, C_proj, D, z.permute(0,2,1), self.dt_proj.bias, delta_softplus=True)
        
        y = y.permute(0, 2, 1) # (B, L, D_in)

        return self.out_proj(y)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

class CrossMamba(nn.Module):
    def __init__(self, dim, d_state=16):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim, d_state=d_state, bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, 128, 128)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi

class SingleMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim, d_state=d_state, bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
class TokenSwapMamba(nn.Module):
    def __init__(self, dim, d_state=16):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim, d_state=d_state, bimamba_type=None)
        self.panencoder = Mamba(dim, d_state=d_state, bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan,ms_residual,pan_residual):
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual
