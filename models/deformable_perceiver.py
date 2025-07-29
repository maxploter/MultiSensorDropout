import copy

from torch import nn
import torch

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from models.ops.modules import MSDeformAttn

from models.backbone import build_backbone
from models.perceiver import CenterPointDetectionHead, ObjectDetectionHead


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class DeformablePerceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        n_levels = 4,
        n_points = 4
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
          n_levels: Number of feature levels for deformable attention.
          n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.to_latent = nn.Linear(input_dim, latent_dim)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: MSDeformAttn(latent_dim, n_levels, cross_heads, n_points)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        fourier_channels = 2 * ((num_freq_bands * 2) + 1)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, fourier_channels)) # TODO

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

        input_proj_list = []
        backbone_num_channels = [64, 128, 256] #TODO
        for _ in range(n_levels):
            in_channels = backbone_num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, latent_dim, kernel_size=1),
                nn.GroupNorm(32, latent_dim),
            ))

        self.input_proj = nn.ModuleList(input_proj_list)
        self.reference_points = nn.Linear(latent_dim, 2)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)



    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        data,
        latents=None,  # (b, num_latents, latent_dim)
        mask = None,
        return_embeddings = True,
        **kwargs
    ):
        pos_embeds = []
        srcs = []
        masks = []

        for l, feat in enumerate(data):
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            srcs.append(self.input_proj[l](feat))

            mask = torch.zeros_like(feat[:, 0, :, :], dtype=torch.bool)
            masks.append(mask)

            feat = feat.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C
            b, *axis, _, device, dtype = *feat.shape, feat.device, feat.dtype
            assert len(axis) == self.input_axis, 'input data must have the right number of axis'

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)
            pos_embeds.append(enc_pos)
            # data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        # prepare input for encoder

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        src = self.to_latent(torch.cat((src_flatten, lvl_pos_embed_flatten), dim = -1))


        if latents is not None:
            x = latents
        else:
            x = repeat(self.latents, 'n d -> b n d', b=b)

        reference_points = self.reference_points(x).sigmoid()

        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] \
                                     * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(
                query=x,
                reference_points=reference_points_input,
                input_flatten=src,
                input_spatial_shapes=spatial_shapes,
                input_level_start_index=level_start_index,
                # input_padding_mask = mask_flatten
            ) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

def build_model_deformable_perceiver(args, num_classes, input_image_view_size):
    gh, gw = args.grid_size

    backbone = build_backbone(args, input_image_view_size=input_image_view_size)

    num_freq_bands = args.num_freq_bands
    fourier_channels = 2 * ((num_freq_bands * 2) + 1)

    num_channels_from_backbone = backbone.num_channels

    gh, gw = args.grid_size

    num_sensors = gh * gw

    num_channels = backbone.num_channels

    n_levels = 3
    n_points = 4 #TODO

    perceiver = DeformablePerceiver(
        input_channels=args.hidden_dim,  # number of channels for each token of the input
        input_axis=2,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=num_freq_bands,  # number of freq bands, with original value (2 * K + 1)
        max_freq=args.max_freq,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=args.enc_layers,  # depth of net. The shape of the final attention mechanism will be:
        #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents=args.num_queries,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=args.hidden_dim,  # latent dimension
        cross_heads=args.enc_nheads_cross,  # number of heads for cross attention. paper said 1
        latent_heads=args.nheads,  # number of heads for latent self attention, 8
        cross_dim_head=(num_channels + fourier_channels) // args.enc_nheads_cross,
        # number of dimensions per cross attention head
        latent_dim_head=args.hidden_dim // args.nheads,  # number of dimensions per latent self attention head
        num_classes=-1,  # NOT USED. output number of classes.
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
        weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data=True,
        # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn=args.self_per_cross_attn,  # number of self attention blocks per cross attention
        final_classifier_head=False,  # mean pool and project embeddings to number of classes (num_classes) at the end
        n_levels=n_levels,  # number of feature levels for deformable attention
        n_points=n_points  # number of sampling points per attention head per feature level
    )


    if args.object_detection:
        classification_heads = ObjectDetectionHead(
            num_classes=num_classes,
            latent_dim=args.hidden_dim
        )
    else:
        classification_heads = CenterPointDetectionHead(
            num_classes=num_classes,
            latent_dim=args.hidden_dim
        )

    return backbone, nn.Identity(), perceiver, classification_heads