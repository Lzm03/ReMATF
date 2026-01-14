import random
from pdb import set_trace as stx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision.ops import DeformConv2d

from .DefConv import DeformConv3d
from .nnMamba import nnMambaSeg, nnMambaRefine
from .SwinUnet_3D import swinUnet_t_3D


##########################################################################
## Layer Norm
class LayerNorm3D(nn.Module):

    def __init__(self, dim, elementwise_affine=True, bias=True):
        super(LayerNorm3D, self).__init__()
        self.LN = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

    def to_3d(self, x):
        return rearrange(x, "b c t h w -> b (t h w) c")

    def to_5d(self, x, t, h, w):
        return rearrange(x, "b (t h w) c -> b c t h w", t=t, h=h, w=w)

    def forward(self, x):
        t, h, w = x.shape[-3:]
        return self.to_5d(self.LN(self.to_3d(x)), t, h, w)


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = LayerNorm3D(dim, bias=False)
        else:
            self.body = LayerNorm3D(dim)

    def forward(self, x):
        return self.body(x)


class Deconv3D_Block(nn.Module):

    def __init__(
        self,
        inp_feat,
        out_feat,
        kernel=3,
        stride=2,
        padding=1,
        norm=LayerNorm3D,
        conv_type="dw",
    ):
        super(Deconv3D_Block, self).__init__()
        if conv_type == "normal":
            self.deconv = nn.Sequential(
                norm(inp_feat),
                nn.ConvTranspose3d(
                    inp_feat,
                    out_feat,
                    kernel_size=(kernel, kernel, kernel),
                    stride=(1, stride, stride),
                    padding=(padding, padding, padding),
                    output_padding=(0, 1, 1),
                    bias=True,
                ),
                nn.GELU(),
            )
        if conv_type == "dw":
            self.deconv = nn.Sequential(
                norm(inp_feat),
                nn.ConvTranspose3d(
                    inp_feat,
                    inp_feat,
                    kernel_size=(kernel, kernel, kernel),
                    stride=(1, stride, stride),
                    padding=(padding, padding, padding),
                    output_padding=(0, 1, 1),
                    groups=inp_feat,
                    bias=True,
                ),
                nn.Conv3d(in_channels=inp_feat, out_channels=out_feat, kernel_size=1),
                nn.GELU(),
            )

    def forward(self, x):
        return self.deconv(x)


# def TiltWarp(
#     x,
#     flow,
#     interp_mode="bilinear",
#     padding_mode="zeros",
#     align_corners=True,
#     use_pad_mask=False,
# ):
#     """_summary_

#     Args:
#         x (tensor): input tensor
#         flow (tensor): input flow tensor
#         interp_mode (str, optional): interpolation mode. Defaults to "bilinear".
#         padding_mode (str, optional): padding mode. Defaults to "zeros".
#         align_corners (bool, optional): align corners. Defaults to True.
#         use_pad_mask (bool, optional): use padding mask. Defaults to False.

#     Returns:
#         tensor: output warped tensor
#     """

#     _, n, c, h, w = x.size()
#     x = x.reshape((-1, c, h, w))

#     flow = flow.permute(0, 2, 3, 4, 1).reshape((-1, h, w, 2))
#     # create mesh grid
#     grid_y, grid_x = torch.meshgrid(
#         torch.arange(0, h, dtype=x.dtype, device=x.device),
#         torch.arange(0, w, dtype=x.dtype, device=x.device),
#     )
#     grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#     grid.requires_grad = False
#     vgrid = grid + flow

#     vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
#     vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
#     vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#     output = F.grid_sample(
#         x,
#         vgrid_scaled,
#         mode=interp_mode,
#         padding_mode=padding_mode,
#         align_corners=align_corners,
#     )
#     output = output.reshape((-1, n, c, h, w))
#     return output


def TiltWarp(
    x,
    flow,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    use_pad_mask=False,
):
    B, T, C, H, W = x.size()
    x = x.reshape(B * T, C, H, W)

    flow = flow.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, 2)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=x.dtype, device=x.device),
        torch.arange(W, dtype=x.dtype, device=x.device),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B * T, 1, 1, 1)

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return output.reshape(B, T, C, H, W)


class DWconv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(DWconv3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype,
            ),
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return self.conv(x)


class DeformConv3D_Block(nn.Module):

    def __init__(
        self, inp_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DeformConv3D_Block, self).__init__()
        self.deform_conv = DeformConv3d(
            inp_feat,
            out_feat,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3D_Block(nn.Module):
    def __init__(
        self,
        inp_feat,
        out_feat,
        kernel=3,
        stride=1,
        padding=1,
        norm=nn.BatchNorm3d,
        conv_type="normal",
        residual=None,
    ):
        super(Conv3D_Block, self).__init__()

        if conv_type == "normal":
            conv3d = nn.Conv3d
        elif conv_type == "dw":
            conv3d = DWconv3D
        elif conv_type == "deform":
            conv3d = DeformConv3D_Block

        self.conv1 = nn.Sequential(
            conv3d(
                inp_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            norm(out_feat),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            conv3d(
                out_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            norm(out_feat),
            nn.LeakyReLU(),
        )

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = conv3d(
                inp_feat, out_feat, kernel_size=1, bias=False
            )

    def forward(self, x):
        res = x
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class FeatureAlign(nn.Module):

    def __init__(self, inp_feat, embed_feat):
        super(FeatureAlign, self).__init__()

        self.conv_blk = DeformConv3D_Block(
            inp_feat, embed_feat, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.out = nn.Conv3d(
            embed_feat, 2, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x):
        flow = self.out(self.conv_blk(x))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        out = TiltWarp(x, flow)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out, flow


class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, flow_dim_ratio, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.dim = dim
        self.flow_dim = int(dim * flow_dim_ratio)
        self.project_in = nn.Conv3d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv3d(
            hidden_features, dim + self.flow_dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if self.flow_dim == 0:
            x = self.project_out(x)
            return x
        else:
            x, f = self.project_out(x).split([self.dim, self.flow_dim], dim=1)
            return x, f


class AttentionCTSF(nn.Module):

    def __init__(self, dim, num_heads, bias, n_frames=10):
        super(AttentionCTSF, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        shuffle_g = 8
        # shuffle_g = 16
        self.get_qkv = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias),
            nn.Conv3d(
                dim * 3,
                dim * 3,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                groups=dim * 3,
                bias=bias,
            ),
            Rearrange("b (c1 c2) t h w -> b c2 h w (c1 t)", c1=shuffle_g),
            nn.Linear(n_frames * shuffle_g, n_frames * shuffle_g),
            Rearrange("b c2 h w (c1 t) -> b (c2 c1 t) h w", c1=shuffle_g),
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, t, h, w = x.shape

        qkv = self.get_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q, "b (head c t) h w -> b head (c t) (h w)", head=self.num_heads, t=t
        )
        k = rearrange(
            k, "b (head c t) h w -> b head (c t) (h w)", head=self.num_heads, t=t
        )
        v = rearrange(
            v, "b (head c t) h w -> b head (c t) (h w)", head=self.num_heads, t=t
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            "b head (c t) (h w) -> b (head c) t h w",
            head=self.num_heads,
            t=t,
            h=h,
            w=w,
        )
        out = self.project_out(out)
        return out
    
    
class TurbCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(TurbCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, 
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.proj_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, turb_emb):
        B, C, T, H, W = x.shape
        N = T * H * W

        feat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)

        q = self.q_proj(turb_emb).unsqueeze(1)
        k = self.k_proj(feat) 
        v = self.v_proj(feat)     

        def split_heads(t, is_q=False):
            if is_q:
                return t.view(B, 1, self.num_heads, C // self.num_heads).transpose(1, 2) 
            else:
                return t.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2) 

        q = split_heads(q, is_q=True)
        k = split_heads(k, is_q=False) 
        v = split_heads(v, is_q=False) 

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, 1, C)  
        out = self.proj_out(out).squeeze(1)      

        out = out.view(B, C, 1, 1, 1)
        return x + out



class TransformerBlock(nn.Module):
    """Transformer Block

    Args:
        (nn.Module): torch nn base class
    """

    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LN_bias,
        flow_dim_ratio,
        att_ckpt,
        ffn_ckpt,
        n_frames=10,
    ):
        super(TransformerBlock, self).__init__()
        self.flow_dim_ratio = flow_dim_ratio
        self.att_ckpt = att_ckpt
        self.ffn_ckpt = ffn_ckpt

        self.norm1 = LayerNorm3D(dim, bias=LN_bias)
        self.attn = AttentionCTSF(dim, num_heads, bias, n_frames)

        self.norm2 = LayerNorm3D(dim, bias=LN_bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, flow_dim_ratio, bias)

    def forward(self, x):
        x = self.norm1(x)
        if self.att_ckpt:
            x = x + checkpoint.checkpoint(self.attn, x)
        else:
            x = x + self.attn(x)

        x = self.norm2(x)
        if self.ffn_ckpt:
            o = checkpoint.checkpoint(self.ffn, x)
        else:
            o = self.ffn(x)

        if self.flow_dim_ratio > 0:
            return x + o[0], o[1]
        else:
            return x + o


class Preprocessing(nn.Module):

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(Preprocessing, self).__init__()
        self.proj = nn.Conv3d(
            in_c,
            embed_dim,
            kernel_size=(3, 7, 7),
            stride=1,
            padding=(1, 3, 3),
            bias=bias,
        )

    def forward(self, x):
        """forward pass

        Args:
            x (tensor): input tensor

        Returns:
            tensor: output tensor
        """
        x = self.proj(x)
        return x


class Downsample(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv3d(
                dim_in, dim_out // 4, kernel_size=3, stride=1, padding=1, bias=False
            ),
            Rearrange("b c t (h rh) (w rw) -> b (c rh rw) t h w", rh=2, rw=2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv3d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            Rearrange("b (c rh rw) t h w -> b c t (h rh) (w rw)", rh=2, rw=2),
        )

    def forward(self, x):
        return self.body(x)


def add_noise(img, sigma):
    noise = (sigma**0.5) * torch.randn(img.shape, device=img.device)
    out = img + noise
    return out.clamp(0, 1)


class UNet3D(nn.Module):

    def __init__(
        self,
        num_channels=3,
        feat_channels=[32, 128, 128, 256],
        norm="BN",
        conv_type="normal",
        residual="conv",
        noise=0,
    ):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals
        super(UNet3D, self).__init__()

        # Encoder downsamplers
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        # Encoder convolutions
        if norm == "BN":
            norm3d = nn.BatchNorm3d
        elif norm == "LN":
            norm3d = LayerNorm3D

        self.conv_blk1 = Conv3D_Block(
            num_channels,
            feat_channels[0],
            kernel=7,
            stride=1,
            padding=3,
            norm=norm3d,
            conv_type="normal",
            residual=residual,
        )
        self.conv_blk2 = Conv3D_Block(
            feat_channels[0],
            feat_channels[1],
            kernel=7,
            stride=1,
            padding=3,
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )
        self.conv_blk3 = Conv3D_Block(
            feat_channels[1],
            feat_channels[2],
            kernel=5,
            stride=1,
            padding=2,
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )
        self.conv_blk4 = Conv3D_Block(
            feat_channels[2],
            feat_channels[3],
            kernel=3,
            stride=1,
            padding=1,
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )

        # Decoder convolutions
        self.dec_conv_blk3 = Conv3D_Block(
            2 * feat_channels[2],
            feat_channels[2],
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )
        self.dec_conv_blk2 = Conv3D_Block(
            2 * feat_channels[1],
            feat_channels[1],
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )
        self.dec_conv_blk1 = Conv3D_Block(
            2 * feat_channels[0],
            feat_channels[0],
            norm=norm3d,
            conv_type=conv_type,
            residual=residual,
        )

        # Decoder upsamplers
        self.deconv_blk3 = Deconv3D_Block(
            feat_channels[3], feat_channels[2], conv_type=conv_type
        )
        self.deconv_blk2 = Deconv3D_Block(
            feat_channels[2], feat_channels[1], conv_type=conv_type
        )
        self.deconv_blk1 = Deconv3D_Block(
            feat_channels[1], feat_channels[0], conv_type=conv_type
        )

        # Final 1*1 Conv Segmentation map
        self.out_conv3 = nn.Conv3d(
            feat_channels[2], 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.out_conv2 = nn.Conv3d(
            feat_channels[1], 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.out_conv1 = nn.Conv3d(
            feat_channels[0], 2, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.upsample3 = nn.Upsample(
            size=None, scale_factor=(1, 4, 4), mode="trilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            size=None, scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
        )
        self.noise = noise

    def forward(self, x):
        """forward pass

        Args:
            x (tensor): input tensor

        Returns:
            tensor: output tensor
        """
        # Encoder part
        xin = x.permute(0, 2, 1, 3, 4)
        xin = add_noise(xin, self.noise * random.random())
        x1 = self.conv_blk1(xin)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        base = self.conv_blk4(x_low3)

        # Decoder part

        d3 = torch.cat([self.deconv_blk3(base), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        flow3 = self.out_conv3(d_high3)
        flow2 = self.out_conv2(d_high2)
        flow1 = self.out_conv1(d_high1)
        out_3 = TiltWarp(x, self.upsample3(flow3))
        out_2 = TiltWarp(out_3, self.upsample2(flow2))
        out = TiltWarp(out_2, flow1)
        return out


def make_level_blk(
    dim,
    num_tb,
    nhead,
    ffn,
    bias,
    LN_bias,
    flow_dim_ratio=0,
    att_ckpt=False,
    ffn_ckpt=False,
    n_frames=10,
    self_align=True,
):
    module = []
    for i in range(num_tb - 1):
        module.append(
            TransformerBlock(
                dim=dim,
                num_heads=nhead,
                ffn_expansion_factor=ffn,
                bias=bias,
                LN_bias=LN_bias,
                flow_dim_ratio=0,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
            )
        )
        if i == int(num_tb / 2) and self_align:
            module.append(FeatureAlign(dim, dim))

    module.append(
        TransformerBlock(
            dim=dim,
            num_heads=nhead,
            ffn_expansion_factor=ffn,
            bias=bias,
            LN_bias=LN_bias,
            flow_dim_ratio=flow_dim_ratio,
            att_ckpt=att_ckpt,
            ffn_ckpt=ffn_ckpt,
            n_frames=n_frames,
        )
    )
    # return module
    return nn.ModuleList(module)


class TMT_MS(nn.Module):
    """Multiscale TMT

    Args:
        (nn.Module): torch nn base class
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LN_bias=True,
        warp_mode="none",
        n_frames=10,
        out_residual=True,
        att_ckpt=False,
        ffn_ckpt=False,
        swindef=False,
        mambadef=False,
        deform3D=False,
    ):

        super(TMT_MS, self).__init__()
        self.deform3D = deform3D
        self.swindef = swindef
        self.mambadef = mambadef
        if warp_mode == "enc":
            align = [True, True, True, False, False, False, False]
        elif warp_mode == "dec":
            align = [False, False, False, True, True, True, True]
        elif warp_mode == "all":
            align = [True, True, True, True, True, True, True]
        else:
            align = [False, False, False, False, False, False, False]

        self.out_residual = out_residual
        self.use_turb_cond = True
        self.turb_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.turb_cross_attn = TurbCrossAttention(dim=dim, num_heads=heads[0], bias=bias)

        self.getFeature1 = Preprocessing(inp_channels, dim)
        self.getFeature2 = Preprocessing(inp_channels, dim)
        self.getFeature3 = Preprocessing(inp_channels, dim)

        self.encode_l1 = nn.Sequential(
            *make_level_blk(
                dim=dim,
                num_tb=num_blocks[0],
                nhead=heads[0],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[0],
            )
        )

        self.down1_2 = Downsample(
            dim, int(dim * 2**1) - dim
        )  ## From Level 1 to Level 2
        self.encode_l2 = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**1),
                num_tb=num_blocks[1],
                nhead=heads[1],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[1],
            )
        )

        self.down2_3 = Downsample(
            int(dim * 2**1), int(dim * 2**2) - dim
        )  ## From Level 2 to Level 3
        self.encode_l3 = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**2),
                num_tb=num_blocks[2],
                nhead=heads[2],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[2],
            )
        )

        self.down3_4 = Downsample(
            int(dim * 2**2), int(dim * 2**3)
        )  ## From Level 3 to Level 4
        self.embedding = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**3),
                num_tb=num_blocks[3],
                nhead=heads[3],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[3],
            )
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_l3 = nn.Conv3d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decode_l3 = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**2),
                num_tb=num_blocks[2],
                nhead=heads[2],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[4],
            )
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_l2 = nn.Conv3d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decode_l2 = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**1),
                num_tb=num_blocks[1],
                nhead=heads[1],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[5],
            )
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decode_l1 = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**1),
                num_tb=num_blocks[0],
                nhead=heads[0],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[6],
            )
        )

        self.refinement = nn.Sequential(
            *make_level_blk(
                dim=int(dim * 2**1),
                num_tb=num_refinement_blocks,
                nhead=heads[0],
                ffn=ffn_expansion_factor,
                bias=bias,
                LN_bias=LN_bias,
                att_ckpt=att_ckpt,
                ffn_ckpt=ffn_ckpt,
                n_frames=n_frames,
                self_align=align[6],
            )
        )

        self.output = nn.Conv3d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        if self.swindef:
            self.swin3d = swinUnet_t_3D(in_channel=1, hidden_dim=32, num_classes=1)
            self.swin_conv_in = nn.ConvTranspose3d(
                in_channels=dim,
                out_channels=1,
                kernel_size=(11, 1, 1),
                stride=(11, 1, 1),
                padding=(2, 0, 0),
            )
            self.swin_conv_out = nn.Conv3d(
                in_channels=1,
                out_channels=dim * 2,
                kernel_size=(3, 1, 1),
                stride=(11, 1, 1),
                padding=(2, 0, 0),
            )

        if self.mambadef:
            # self.mamba_refine = nnMambaRefine(in_ch=dim*2, hidden_ch=dim*2, blocks=2)
            self.mamba_refine = nnMambaRefine(in_ch=dim*8, hidden_ch=dim*8, blocks=2)
            # self.mamba_conv_in_3d  = nn.Conv3d(dim, 1, kernel_size=1, stride=1, padding=0)
            # self.mamba_conv_out_3d = nn.Conv3d(1, dim*2, kernel_size=1, stride=1, padding=0)


        if self.deform3D:
            self.unet3d = UNet3D(norm="LN", residual="pool", conv_type="dw")
            
    def forward_with_optional_flow(self, module):
        def forward_fn(x):
            flows = []
            for blk in module:
                if isinstance(blk, FeatureAlign):
                    x, flow = blk(x)
                    flows.append(flow)
                else:
                    x = blk(x)
            return x, flows
        return forward_fn

    # def forward(self, inp_img):
    #     """forward pass

    #     Args:
    #         inp_img (tensor): input tensor

    #     Returns:
    #         tensor: output tensor
    #     """
        
    #     flows = []

    #     if self.deform3D:
    #         # Definitely better to use deformable feature alignment. 
    #         inp_img = self.unet3d(inp_img.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

    #     if self.mambadef:
    #         out_dec_l1 = self.getFeature1(inp_img)
    #         out_dec_l1 = self.mamba_conv_in_3d(out_dec_l1)
    #         out_dec_l1 = self.mamba3d(out_dec_l1)
    #         out_dec_l1 = self.mamba_conv_out_3d(out_dec_l1)

    #     if self.swindef:
    #         out_dec_l1 = self.getFeature1(inp_img)
    #         out_dec_l1 = self.swin_conv_in(out_dec_l1)
    #         out_dec_l1 = self.swin3d(out_dec_l1)
    #         out_dec_l1 = self.swin_conv_out(out_dec_l1)
        
    #     if not (self.swindef or self.mambadef): 
    #         _, _, t, h, w = inp_img.shape
    #         inp_img2 = F.interpolate(
    #             inp_img, size=(t, h // 2, w // 2), mode="trilinear", align_corners=False
    #         )
    #         inp_img3 = F.interpolate(
    #             inp_img, size=(t, h // 4, w // 4), mode="trilinear", align_corners=False
    #         )
    #         inp_enc_l1 = self.getFeature1(inp_img)
    #         inp_enc_l2 = self.getFeature2(inp_img2)
    #         inp_enc_l3 = self.getFeature3(inp_img3)

    #         # out_enc_l1 = self.encode_l1(inp_enc_l1)
    #         # out_enc_l2 = self.encode_l2(
    #         #     torch.cat([inp_enc_l2, self.down1_2(out_enc_l1)], 1)
    #         # )
    #         # out_enc_l3 = self.encode_l3(
    #         #     torch.cat([inp_enc_l3, self.down2_3(out_enc_l2)], 1)
    #         # )
            
    #         out_enc_l1, _ = self.forward_with_optional_flow(self.encode_l1)(inp_enc_l1)
    #         out_enc_l2, _ = self.forward_with_optional_flow(self.encode_l2)(torch.cat([inp_enc_l2, self.down1_2(out_enc_l1)], 1))
    #         out_enc_l3, _ = self.forward_with_optional_flow(self.encode_l3)(torch.cat([inp_enc_l3, self.down2_3(out_enc_l2)], 1))

    #         inp_enc_l4 = self.down3_4(out_enc_l3)
    #         # embedding = self.embedding(inp_enc_l4)
    #         embedding, _ = self.forward_with_optional_flow(self.embedding)(inp_enc_l4)

    #         inp_dec_l3 = self.up4_3(embedding)
    #         inp_dec_l3 = torch.cat([inp_dec_l3, out_enc_l3], 1)
    #         inp_dec_l3 = self.reduce_chan_l3(inp_dec_l3)
    #         # out_dec_l3 = self.decode_l3(inp_dec_l3)
    #         out_dec_l3, _ = self.forward_with_optional_flow(self.decode_l3)(inp_dec_l3)

    #         inp_dec_l2 = self.up3_2(out_dec_l3)
    #         inp_dec_l2 = torch.cat([inp_dec_l2, out_enc_l2], 1)
    #         inp_dec_l2 = self.reduce_chan_l2(inp_dec_l2)
    #         # out_dec_l2 = self.decode_l2(inp_dec_l2)
    #         out_dec_l2, _ = self.forward_with_optional_flow(self.decode_l2)(inp_dec_l2)

    #         inp_dec_l1 = self.up2_1(out_dec_l2)
    #         inp_dec_l1 = torch.cat([inp_dec_l1, out_enc_l1], 1)
    #         # out_dec_l1 = self.decode_l1(inp_dec_l1)
    #         out_dec_l1, _ = self.forward_with_optional_flow(self.decode_l1)(inp_dec_l1)

    #     # out_dec_l1 = self.refinement(out_dec_l1)
    #     out_dec_l1, _ = self.forward_with_optional_flow(self.refinement)(out_dec_l1)

    #     if self.out_residual:
    #         out = self.output(out_dec_l1) + inp_img
    #     else:
    #         out = self.output(out_dec_l1)
    #     return out
    
    def forward(self, inp_img, turb_level=None):
        
        """forward pass

        Args:
            inp_img (tensor): input tensor

        Returns:
            tensor: output tensor
        """

        flows = []

        if self.deform3D:
            inp_img = self.unet3d(inp_img.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        elif self.swindef:
            out_dec_l1 = self.getFeature1(inp_img)
            out_dec_l1 = self.swin_conv_in(out_dec_l1)
            out_dec_l1 = self.swin3d(out_dec_l1)
            out_dec_l1 = self.swin_conv_out(out_dec_l1)

        else:
            _, _, t, h, w = inp_img.shape
            inp_img2 = F.interpolate(inp_img, size=(t, h // 2, w // 2), mode="trilinear", align_corners=False)
            inp_img3 = F.interpolate(inp_img, size=(t, h // 4, w // 4), mode="trilinear", align_corners=False)

            inp_enc_l1 = self.getFeature1(inp_img)
            
            if self.use_turb_cond and (turb_level is not None):
                if isinstance(turb_level, (float, int)):
                    turb_level = torch.tensor(
                        [turb_level],
                        dtype=inp_enc_l1.dtype,
                        device=inp_enc_l1.device
                    )  # [1]
                if turb_level.dim() == 0:   # 标量 tensor
                    turb_level = turb_level.unsqueeze(0)        # [1]
                if turb_level.dim() == 1:   # [B]
                    turb_level = turb_level.unsqueeze(1)        # [B,1]

                turb_level = turb_level.to(inp_enc_l1.device, dtype=inp_enc_l1.dtype)  # [B,1]
                B = inp_enc_l1.shape[0]
                assert turb_level.shape[0] == B, "turb_level batch size mismatch"

                turb_emb = self.turb_embed(turb_level).view(B, -1)   # [B, dim]
                # Cross-attention: turb_emb 作为 query，全局调制 L1 特征
                inp_enc_l1 = self.turb_cross_attn(inp_enc_l1, turb_emb)
            
            inp_enc_l2 = self.getFeature2(inp_img2)
            inp_enc_l3 = self.getFeature3(inp_img3)

            out_enc_l1, f = self.forward_with_optional_flow(self.encode_l1)(inp_enc_l1); flows += f
            out_enc_l2, f = self.forward_with_optional_flow(self.encode_l2)(torch.cat([inp_enc_l2, self.down1_2(out_enc_l1)], 1)); flows += f
            out_enc_l3, f = self.forward_with_optional_flow(self.encode_l3)(torch.cat([inp_enc_l3, self.down2_3(out_enc_l2)], 1)); flows += f
            embedding, f = self.forward_with_optional_flow(self.embedding)(self.down3_4(out_enc_l3)); flows += f
            
            if self.mambadef:
                refine_out = self.mamba_refine(embedding)
                embedding = embedding + 0.2 * refine_out

            inp_dec_l3 = self.up4_3(embedding)
            inp_dec_l3 = torch.cat([inp_dec_l3, out_enc_l3], 1)
            inp_dec_l3 = self.reduce_chan_l3(inp_dec_l3)
            out_dec_l3, f = self.forward_with_optional_flow(self.decode_l3)(inp_dec_l3); flows += f

            inp_dec_l2 = self.up3_2(out_dec_l3)
            inp_dec_l2 = torch.cat([inp_dec_l2, out_enc_l2], 1)
            inp_dec_l2 = self.reduce_chan_l2(inp_dec_l2)
            out_dec_l2, f = self.forward_with_optional_flow(self.decode_l2)(inp_dec_l2); flows += f

            inp_dec_l1 = self.up2_1(out_dec_l2)
            inp_dec_l1 = torch.cat([inp_dec_l1, out_enc_l1], 1)
            out_dec_l1, f = self.forward_with_optional_flow(self.decode_l1)(inp_dec_l1); flows += f

        out_dec_l1, f = self.forward_with_optional_flow(self.refinement)(out_dec_l1); flows += f

        if self.out_residual:
            out = self.output(out_dec_l1) + inp_img
        else:
            out = self.output(out_dec_l1)

        return out, flows
