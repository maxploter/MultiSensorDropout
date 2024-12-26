from functools import wraps
from math import pi

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import Tensor
from torch import nn, einsum

from models.backbone import build_backbone


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(mask, max_neg_value)  # Fills elements of self tensor with value where mask is True

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Perceiver(nn.Module):
    def __init__(
            self,
            *,
            num_freq_bands,
            depth,
            max_freq,
            input_channels=3,
            input_axis=2,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1000,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=1,
            final_classifier_head=True
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
        """
        for param_name, param_value in locals().items():
            if param_name != 'self':
                print(f"{param_name}: {param_value}")
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
            self,
            data,  # b ()
            latents=None,  # (b, num_latents, latent_dim)
            mask=None,
            return_embeddings=False
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        if latents is not None:
            x = latents
        else:
            x = repeat(self.latents, 'n d -> b n d', b=b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x_cross = cross_attn(x, context=data, mask=mask)

            if mask is not None:
                # Check which rows (batch-wise) have all elements equal to 1 (fully masked)
                # mask_all_ones will be a [batch_size] tensor of True/False
                mask_all_ones = mask.view(mask.size(0), -1).all(dim=1)

                mask_all_ones = mask_all_ones.view(-1, 1, 1)

                # Zero out the corresponding rows in tgt2
                x_cross = x_cross.masked_fill(mask_all_ones, float(0))

            x = x_cross + x

            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)


class PerceiverDetection(nn.Module):

    def __init__(self, backbone, perceiver, classification_head):
        super().__init__()
        self.backbone = backbone
        self.perceiver = perceiver
        self.classification_head = classification_head
        # Compatibility with TrackingBaseModel
        self.num_queries = perceiver.latents.shape[0]
        self.hidden_dim = perceiver.latents.shape[1]
        self.overflow_boxes = False

    def forward(self, samples, targets: list = None, latents: Tensor = None):

        src = self.backbone(samples)
        src = src.permute(0, 2, 3, 1)

        hs = self.perceiver(
            data=src,
            return_embeddings=True,
            latents=latents
        )
        out = self.classification_head(hs)

        # TODO: double check if normilization should be disabled
        out['hs_embed'] = hs

        return (
            out,
            targets,
            None,
            None,  # Memory, is an output from encoder
            hs
        )


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes, num_latents, latent_dim):
        """ Initializes the model.
        Parameters:
            num_classes: number of object classes
            num_latents: number of object queries, ie detection slot. This is the maximal number of objects
                         model can detect in a single image. For COCO, we recommend 100 queries.
            latent_dim: dimension of the latent object query.
        """
        super().__init__()
        self.num_queries = num_latents
        self.class_embed = nn.Linear(latent_dim, num_classes + 1)
        self.center_points_embed = MLP(latent_dim, latent_dim, 2, 3)

    def forward(self, hs: Tensor):
        """Forward pass of the ObjectDetectionHead.
            Parameters:
                - hs: Tensor
                    Hidden states from the model, of shape [batch_size x num_queries x latent_dim].

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        outputs_class = self.class_embed(hs)
        outputs_coord = self.center_points_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_center_points': outputs_coord}
        return out


def build_model_perceiver(args, num_classes):

    backbone = build_backbone(args)

    num_freq_bands = args.num_freq_bands
    fourier_channels = 2 * ((num_freq_bands * 2) + 1)

    num_queries = args.num_objects
    num_channels = backbone.num_channels

    perceiver = Perceiver(
        input_channels=num_channels,  # number of channels for each token of the input
        input_axis=2,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=num_freq_bands,  # number of freq bands, with original value (2 * K + 1)
        max_freq=args.max_freq,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=args.enc_layers,  # depth of net. The shape of the final attention mechanism will be:
        #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents=num_queries,
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
        final_classifier_head=False  # mean pool and project embeddings to number of classes (num_classes) at the end
    )

    classifier_head = ObjectDetectionHead(
        num_classes=num_classes,
        num_latents=num_queries,
        latent_dim=args.hidden_dim
    )

    return backbone, perceiver, classifier_head
