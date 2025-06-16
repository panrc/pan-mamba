import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from .refine import Refine
from .mamba_module import Mamba, LayerNorm, CrossMamba, SingleMambaBlock, TokenSwapMamba
import time

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms

class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x

class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        d_state = args['data']['d_state']
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.deep_fusion1= CrossMamba(self.embed_dim, d_state=d_state)
        self.deep_fusion2 = CrossMamba(self.embed_dim, d_state=d_state)
        self.deep_fusion3 = CrossMamba(self.embed_dim, d_state=d_state)
        self.deep_fusion4 = CrossMamba(self.embed_dim, d_state=d_state)
        self.deep_fusion5 = CrossMamba(self.embed_dim, d_state=d_state)

        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim, d_state=d_state) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim, d_state=d_state) for i in range(8)])
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim, d_state=d_state)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim, d_state=d_state)
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter,4)
    def forward(self,ms,_,pan):

        ms_bic = F.interpolate(ms,scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        # ms_f = ms_bic
        # pan_f = pan
        b,c,h,w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        residual_pan_f = 0
        ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f])
        pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f = self.patchunembe(ms_f,(h,w))
        pan_f = self.patchunembe(pan_f,(h,w))
        ms_f = self.shallow_fusion1(torch.concat([ms_f,pan_f],dim=1))+ms_f
        pan_f = self.shallow_fusion2(torch.concat([pan_f,ms_f],dim=1))+pan_f
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        ms_f,residual_ms_f = self.deep_fusion1(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion3(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion4(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion5(ms_f,residual_ms_f,pan_f)
        ms_f = self.patchunembe(ms_f,(h,w))
        hrms = self.output(ms_f)+ms_bic
        return hrms


