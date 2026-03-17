from torch import nn
import scipy.stats as st
import torch
import numpy as np

import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat


from torch.nn import init as init

import torch.nn.functional as F
from einops import rearrange
import numbers

from functools import partial
from timm.models.layers import DropPath
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.down=nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
                                )

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x
    

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


def make_blocks(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


class DenseLayer(nn.Module):
    """
    Dense layer for residual dense block
    """

    def __init__(self, in_chs, growth_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.conv = conv3x3(in_chs, growth_rate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ResDenseBlock(nn.Module):
    """
    Residual Dense Block
    """

    def __init__(self, in_chs, growth_rate, num_layer, activation='relu'):
        super(ResDenseBlock, self).__init__()
        in_chs_acc = in_chs
        op = []
        for i in range(num_layer):
            op.append(DenseLayer(in_chs_acc, growth_rate, activation))
            in_chs_acc += growth_rate
        self.dense_layers = nn.Sequential(*op)
        self.conv1x1 = conv1x1(in_chs_acc, in_chs)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    """
    Middle network of residual dense blocks
    """

    def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
        self.conv3x3 = conv3x3(in_chs, in_chs)
        self.act = actFunc(activation)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.act(self.conv1x1(out))
        out = self.act(self.conv3x3(out))
        return out


class SpaceToDepth(nn.Module):
    """
    Pixel Unshuffle
    """

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"


# based on https://github.com/rogertrullo/pytorch_convlstm
class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())

def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)

class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class Weight_Fusion(nn.Module):
    def __init__(self, in_planes, ratio=16, L=32, M=2):
        super(Weight_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        self.M = M
        d = max(in_planes // ratio, L)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_planes, d, 1, bias=False), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, in_planes * 2, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # ------------------ 新增 spatial fusion module ------------------
        # 用于产生像素级的空间权重（范围 0..1），以调制通道融合结果
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # 可选：控制空间权重对最终输出影响的系数（0.0~1.0）
        self.spatial_alpha = 1.0
        # ------------------ end新增 ------------------


    def forward(self, x1, x2):
        x = x1 + x2
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out

        out = self.fc1(out)
        out_two = self.fc2(out)

        batch_size = x.size(0)

        out_two = out_two.reshape(batch_size, self.M, self.in_planes, -1)
        # out_two = self.softmax(out_two)
        out_two = self.sigmoid(out_two)

        w_1, w_2 = out_two[:, 0:1, :, :], out_two[:, 1:2, :, :]

        w_1 = w_1.reshape(batch_size, self.in_planes, 1, 1)
        w_2 = w_2.reshape(batch_size, self.in_planes, 1, 1)
        # ORIGINAL:
        # out = w_1 * x1 + w_2 * x2
        # return out

        # ------------------ MODIFIED: channel + spatial decoupled fusion ------------------
        # channel-wise fusion (原有)
        out_channel = w_1 * x1 + w_2 * x2  # B x C x 1 x 1 broadcasted to spatial

        # compute pixel-wise spatial weight map from concatenated features
        spatial_input = torch.cat([x1, x2], dim=1)  # B x (2C) x H x W
        spatial_weight = self.spatial_conv(spatial_input)  # B x C x H x W, in (0,1)

        # fuse: let spatial_weight 控制通道融合与 image 分支的权衡
        # spatial_alpha 控制空间模块影响强度（便于 ablation）
        alpha = float(self.spatial_alpha)
        out = alpha * (spatial_weight * out_channel) + (1.0 - alpha) * ((1 - spatial_weight) * x1 + spatial_weight * x2)

        # Residual: 可保留原始 x1 的一小部分以稳定训练（可选）
        # out = out + 0.1 * x1

        return out
        # ------------------ end MODIFIED ------------------



# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB) MPRNet
class CAB(nn.Module):
    def __init__(self, in_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(in_feat, in_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(in_feat, in_feat, kernel_size, bias=bias))

        self.CA = CALayer(in_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        res = self.CA(x)
        res += x
        return res


class EN_Block(nn.Module):

    def __init__(
        self, in_channels, out_channels, BIN, kernel_size=3, reduction=4, bias=False
    ):
        super(EN_Block, self).__init__()
        self.BIN = BIN
        act = nn.ReLU(inplace=True)
        self.conv = conv(in_channels, out_channels, 3, bias=bias)
        self.CABs = [
            CAB(out_channels, kernel_size, reduction, bias=bias, act=act)
            for _ in range(2)
        ]
        self.CABs = nn.Sequential(*self.CABs)

    def forward(self, x):
        x = self.conv(x)
        x = self.CABs(x)
        return x


class DE_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up = SkipUpSample(in_planes, planes)
        self.decoder = [
            CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)
        ]
        self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)

    def forward(self, x, skpCn):

        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Spatio_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Spatio_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp):

        b, c, h, w = inp.shape

        q = self.q(inp)  # image
        k = self.k(inp)  # event
        v = self.v(inp)  # event

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (k.transpose(-2, -1) @ q) * self.temperature
        attn = attn.softmax(dim=-1)

        return attn, v


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.0) / (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=0)
    return out_filter


class EEC(nn.Module):
    def __init__(self, dim, bias=False):
        super(EEC, self).__init__()

        self.Conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.CA = ChannelAttention(dim)

    def forward(self, f_img, f_event, Mask):

        assert f_img.shape == f_event.shape, "the shape of image doesnt equal to event"
        b, c, h, w = f_img.shape

        F_event = f_event * Mask
        F_event = f_event + F_event
        F_cat = torch.cat([F_event, f_img], dim=1)
        F_conv = self.Conv(F_cat)
        w_c = self.CA(F_conv)
        F_event = F_event * w_c
        F_out = F_event + f_img

        return F_out


class ISC(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type="WithBias"):
        super(ISC, self).__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.SA = Spatio_Attention(dim, num_heads, bias)
        self.CA = ChannelAttention(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2 * dim, dim // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 16, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_img, f_event):

        assert f_img.shape == f_event.shape, "the shape of image doesnt equal to event"
        b, c, h, w = f_img.shape
        SA_att, V = self.SA(f_img)
        F_img = V @ SA_att
        F_img = rearrange(
            F_img, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        CA_att = self.CA(f_img)
        F_img = F_img * CA_att
        F_img = F_img + f_img
        w_i = self.avg_pool(F_img)
        w_e = self.avg_pool(f_event)
        w = torch.cat([w_i, w_e], dim=1)
        w = self.fc2(self.relu1(self.fc1(w)))
        w = self.sigmoid(w)
        F_img = F_img * w
        F_event = f_event * (1 - w)
        F_event = F_event + F_img

        return F_event


class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo."""

    def __init__(self, channels):
        super(Decoder, self).__init__()
        ######Decoder
        self.up1 = DE_Block(channels[3], channels[2])
        self.up2 = DE_Block(channels[2], channels[1])
        self.up3 = DE_Block(channels[1], channels[0])

    def forward(self, input):
        x4 = input[3]
        x3 = self.up1(x4, input[2])
        x2 = self.up2(x3, input[1])
        x1 = self.up3(x2, input[0])
        return x1


# EVSSM


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class LayerNorm1(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# modified
class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # NEW: Explicit dual-frequency branches
        self.low_freq_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.high_freq_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        # self.norm = nn.BatchNorm2d(dim)


    def forward(self, x):
        # ORIGINAL:
        # x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        # x = self.project_out(x)

        # NEW: dual-frequency design
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x_main = F.gelu(x1) * x2
        x_main = self.project_out(x_main)

        # Split into frequency-aware branches
        low_freq = self.low_freq_conv(F.avg_pool2d(x_main, kernel_size=2, stride=2))
        low_freq = F.interpolate(low_freq, size=x_main.shape[-2:], mode="bilinear", align_corners=False)
        high_freq = self.high_freq_conv(x_main - low_freq)  # residual-like high-frequency signal

        # Fuse frequency features
        x = torch.cat([low_freq, high_freq], dim=1)
        x = self.fusion(x)

        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        # x = self.norm(x)
        return x


# EVSSM
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=1, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c h w -> b h w c")
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, "b h w c -> b c h w")

        return out

class CrossMambaFusion(nn.Module):
    def __init__(
        self, 
        dim, 
        d_state=16, 
        d_conv=3, 
        expand=2, 
        dropout=0., 
        bias=False
    ):
        super(CrossMambaFusion, self).__init__()
        
        # 1. 两个独立模态的 LayerNorm
        self.norm_img = LayerNorm(dim, LayerNorm_type="WithBias")
        self.norm_event = LayerNorm(dim, LayerNorm_type="WithBias")

        # 2. 事件分支的 SSM：用于提取纯粹的运动特征
        # 这里的 expand 可以稍微小一点以节省计算量，或者保持一致
        self.event_ssm = SS2D(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            dropout=dropout,
            bias=bias
        )

        # 3. 交叉调制模块 (Cross-Modulation)
        # 将 Event 的特征映射为用于调制 Image 的 Scale (gamma) 和 Shift (beta)
        # 这是一个物理意义很明确的设计：Event 决定 Image 特征的分布
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias),
            nn.SiLU() # 或者 nn.GELU()
        )

        # 4. 图像分支的 SSM：在被 Event 调制后进行去模糊扫描
        self.img_ssm = SS2D(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            dropout=dropout,
            bias=bias
        )
        
        # 5. 最终融合映射
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_img, x_event):
        """
        x_img:   [B, C, H, W] - RGB Image Features
        x_event: [B, C, H, W] - Event Features
        """
        B, C, H, W = x_img.shape
        
        # === Path 1: Process Event to get Motion Context ===
        # Residual connection for event is optional depending on network depth
        feat_event = self.norm_event(x_event)
        
        # Event SS2D 捕捉长距离时空依赖
        feat_event_out = self.event_ssm(feat_event) # 输出 [B, C, H, W]
        
        # === Path 2: Cross-Modal Modulation ===
        # 生成调制参数: split 为 gamma (scale) 和 beta (shift)
        scale, shift = self.cross_interaction(feat_event_out).chunk(2, dim=1)
        
        # === Path 3: Process Image with Motion Guidance ===
        feat_img = self.norm_img(x_img)
        
        # 核心交互：Affine Transformation (AdaIN 风格)
        # 物理含义：Event 强的地方（运动边缘），对 Image 特征进行强化/重加权
        feat_img_modulated = feat_img * (1 + scale) + shift
        
        # Image SS2D 在增强后的特征上进行去模糊
        feat_img_out = self.img_ssm(feat_img_modulated)
        
        # === Path 4: Final Residual Fusion ===
        # 将恢复后的高频信息加回原始特征
        out = x_img + self.proj_out(feat_img_out)
        
        return out

class Cross_EN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, BIN, bias=False):
        super(Cross_EN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        
        # 使用新的 Cross-Mamba 模块
        # 注意：这里假设 CrossMambaFusion 内部处理了输入输出维度保持一致
        self.cm_ss2d_layers = nn.ModuleList([
            CrossMambaFusion(dim=out_channels) for _ in range(BIN)
        ])

    def forward(self, x_img, x_event):
        # 假设这个 Block 现在同时接收 Image 和 Event
        x_img = self.conv(x_img)
        x_event = self.conv(x_event) # 或者保持 Event 维度在外部对齐
        
        for layer in self.cm_ss2d_layers:
            # 每一层 Image 都会被 Event 再次指导
            x_img = layer(x_img, x_event)
            
        return x_img, x_event

class SimpleGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 用 Event 生成 Gate 来调制 Image
        self.gate_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x_img, x_event):
        # Event features generate a gate for Image features
        gate = self.gate_pred(x_event)
        # Apply gate and add residual from Event
        out = self.proj(x_img * gate + x_event * (1 - gate))
        return out

class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        kernel_size=7,
        conv_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        out = x + shortcut
        return rearrange(out, "b h w c -> b c h w")


# EVSSM
##########################################################################
class EVS(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor=3,
        bias=False,
        LayerNorm_type="WithBias",
        att=False,
        idx=3,
        patch=128,
    ):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm1(dim)
            self.attn = SS2D(d_model=dim, patch=patch)
            # self.attn = GatedCNNBlock(dim=dim)

        self.norm2 = LayerNorm1(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.att:

            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class EN_Block0812(nn.Module):

    def __init__(
        self, in_channels, out_channels, BIN, kernel_size=3, reduction=4, bias=False
    ):
        super(EN_Block0812, self).__init__()
        self.BIN = BIN
        act = nn.ReLU(inplace=True)
        self.conv = conv(in_channels, out_channels, 3, bias=bias)
        self.EVSs = [EVS(out_channels, att=True, idx=_) for _ in range(BIN)]
        self.EVSs = nn.Sequential(*self.EVSs)
        # self.CABs = [
        #     CAB(out_channels, kernel_size, reduction, bias=bias, act=act)
        #     for _ in range(2)
        # ]
        # self.CABs = nn.Sequential(*self.CABs)

    def forward(self, x):
        x = self.conv(x)
        x = self.EVSs(x)
        # x = self.CABs(x)
        return x

class TFM4488_27DF_28SCF_18CMF(nn.Module):
    """
    Modified Restoration Network
    - Encoder: Enriched with CrossMambaFusion (CM-SS2D) for event-guided feature learning.
    - Decoder: Standard Dual-UNet decoder with Weight_Fusion.
    """

    def __init__(
        self,
        inChannels_img=3,
        inChannels_event=6,
        outChannels=3,
        ends_with_relu=False,
        channels=None,
        evs_bins=None,
    ):
        super(TFM4488_27DF_28SCF_18CMF, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.channels = list(channels) if channels is not None else [64, 96, 160, 192]
        # Keep BIN values as multiples of 4 to avoid orientation mismatch on non-square inputs.
        self.evs_bins = list(evs_bins) if evs_bins is not None else [4, 4, 8, 8]

        if len(self.channels) != 4:
            raise ValueError(f"channels must have 4 items, got {self.channels}")
        if len(self.evs_bins) != 4:
            raise ValueError(f"evs_bins must have 4 items, got {self.evs_bins}")
        if any((b <= 0 or b % 4 != 0) for b in self.evs_bins):
            raise ValueError(
                f"each evs_bins item must be positive and divisible by 4, got {self.evs_bins}"
            )

        # ---------------------------------------------------------
        # 1. Encoders (保持不变)
        # ---------------------------------------------------------
        self.encoder_img_1 = EN_Block0812(
            inChannels_img, self.channels[0], self.evs_bins[0]
        )
        self.encoder_img_2 = EN_Block0812(
            self.channels[0], self.channels[1], self.evs_bins[1]
        )
        self.encoder_img_3 = EN_Block0812(
            self.channels[1], self.channels[2], self.evs_bins[2]
        )
        self.encoder_img_4 = EN_Block0812(
            self.channels[2], self.channels[3], self.evs_bins[3]
        )

        self.encoder_event_1 = EN_Block0812(
            inChannels_event, self.channels[0], self.evs_bins[0]
        )
        self.encoder_event_2 = EN_Block0812(
            self.channels[0], self.channels[1], self.evs_bins[1]
        )
        self.encoder_event_3 = EN_Block0812(
            self.channels[1], self.channels[2], self.evs_bins[2]
        )
        self.encoder_event_4 = EN_Block0812(
            self.channels[2], self.channels[3], self.evs_bins[3]
        )

        self.down = DownSample()

        # ---------------------------------------------------------
        # 2. Cross-Mamba Fusion (替换掉了 EEC 和 ISC)
        # ---------------------------------------------------------
        # 在每一层特征提取后，让 Event 通过 SS2D 指导 Image 特征
        self.cm_fusion_1 = SimpleGateFusion(self.channels[0]) # 替换为轻量级
        self.cm_fusion_2 = SimpleGateFusion(self.channels[1]) # 替换为轻量级
        # self.cm_fusion_1 = CrossMambaFusion(self.channels[0])
        # self.cm_fusion_2 = CrossMambaFusion(self.channels[1])
        self.cm_fusion_3 = CrossMambaFusion(self.channels[2])
        self.cm_fusion_4 = CrossMambaFusion(self.channels[3])

        # ---------------------------------------------------------
        # 3. Decoders (保持不变)
        # ---------------------------------------------------------
        self.decoder_img_1 = DE_Block(self.channels[3], self.channels[2])
        self.decoder_img_2 = DE_Block(self.channels[2], self.channels[1])
        self.decoder_img_3 = DE_Block(self.channels[1], self.channels[0])

        self.decoder_event_1 = DE_Block(self.channels[3], self.channels[2])
        self.decoder_event_2 = DE_Block(self.channels[2], self.channels[1])
        self.decoder_event_3 = DE_Block(self.channels[1], self.channels[0])

        # ---------------------------------------------------------
        # 4. Original Weight Fusion (保持不变)
        # ---------------------------------------------------------
        self.weight_fusion_1 = Weight_Fusion(self.channels[2])
        self.weight_fusion_2 = Weight_Fusion(self.channels[1])
        self.weight_fusion_3 = Weight_Fusion(self.channels[0])

        self.out = nn.Conv2d(self.channels[0], outChannels, 3, stride=1, padding=1)

    def forward(self, x, event):
        # 移除了 M0, M1 等 mask 计算，因为 CrossMamba 不需要它们

        # ================= Encoder Stage 1 =================
        img_1 = self.encoder_img_1(x)
        event_1 = self.encoder_event_1(event)
        
        # [NEW] Cross-Mamba Interaction
        # 利用 event_1 的运动信息去调制/增强 img_1
        img_1 = self.cm_fusion_1(img_1, event_1)

        down_img_1 = self.down(img_1)
        down_event_1 = self.down(event_1)

        # ================= Encoder Stage 2 =================
        img_2 = self.encoder_img_2(down_img_1)
        event_2 = self.encoder_event_2(down_event_1)

        # [NEW] Cross-Mamba Interaction
        img_2 = self.cm_fusion_2(img_2, event_2)

        down_img_2 = self.down(img_2)
        down_event_2 = self.down(event_2)

        # ================= Encoder Stage 3 =================
        img_3 = self.encoder_img_3(down_img_2)
        event_3 = self.encoder_event_3(down_event_2)

        # [NEW] Cross-Mamba Interaction
        img_3 = self.cm_fusion_3(img_3, event_3)

        down_img_3 = self.down(img_3)
        down_event_3 = self.down(event_3)

        # ================= Encoder Stage 4 (Bottleneck) =================
        img_4 = self.encoder_img_4(down_img_3)
        event_4 = self.encoder_event_4(down_event_3)

        # [NEW] Cross-Mamba Interaction
        # 哪怕在最深层，Event 的全局关联性依然能帮助 Image
        img_4 = self.cm_fusion_4(img_4, event_4)

        # ================= Decoder Stage 1 =================
        # img_4 是已经融合过 Event 信息的 Image 特征
        # img_3 是 Skip Connection (也是融合过的)
        up_img_1 = self.decoder_img_1(img_4, img_3)
        up_event_1 = self.decoder_event_1(event_4, event_3)

        # [Original] Weight Fusion
        fuse_img_1 = self.weight_fusion_1(up_img_1, up_event_1)

        # ================= Decoder Stage 2 =================
        up_img_2 = self.decoder_img_2(fuse_img_1, img_2)
        up_event_2 = self.decoder_event_2(up_event_1, event_2)

        # [Original] Weight Fusion
        fuse_img_2 = self.weight_fusion_2(up_img_2, up_event_2)

        # ================= Decoder Stage 3 =================
        up_img_3 = self.decoder_img_3(fuse_img_2, img_1)
        up_event_3 = self.decoder_event_3(up_event_2, event_1)

        # [Original] Weight Fusion
        de_fuse = self.weight_fusion_3(up_img_3, up_event_3)

        # ================= Output =================
        out = self.out(de_fuse)
        out = out + x # Residual Learning
        
        return out


class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo."""

    def __init__(self, channels):
        super(Decoder, self).__init__()
        ######Decoder
        self.up1 = DE_Block(channels[3], channels[2])
        self.up2 = DE_Block(channels[2], channels[1])
        self.up3 = DE_Block(channels[1], channels[0])

    def forward(self, input):
        x4 = input[3]
        x3 = self.up1(x4, input[2])
        x2 = self.up2(x3, input[1])
        x1 = self.up3(x2, input[0])
        return x1


if __name__ == "__main__":
    from torchinfo import summary

    net = TFM4488_27DF_28SCF_18CMF(3, 6, 3, None).cuda()
    summary(
        net,
        input_size=[(1, 3, 256, 256), (1, 6, 256, 256)],  # 对▒~T img ▒~R~L event
        col_names=["input_size", "output_size", "num_params"],
        verbose=1,
    )
