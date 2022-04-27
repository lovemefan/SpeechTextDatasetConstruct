# -*- coding: utf-8 -*-
# @Time  : 2022/4/7 16:03
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : MixedChunkAttention.py
import math

import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F


class T5RelativePositionBias(nn.Module):
    def __init__(
            self,
            scale,
            causal=False,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
            relative_position,
            causal=True,
            num_buckets=32,
            max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class MixedChunkAttention(nn.Module):
    """
    mixed Local attention per chunk and Global attention across chunks attention form paper
    Transformer Quality in Linear Time https://arxiv.org/pdf/2202.10447.
    and https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py
    """

    def __init__(
            self,
            *,
            dim,
            group_size=256,
            query_dim=128,
            key_dim=128,
            query_key_dim=128,
            expansion_factor=2.,
            causal=False,
            dropout=0.,
            rotary_pos_emb=None,
            norm_klass=nn.LayerNorm,
            shift_tokens=False
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal=causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_q = nn.Sequential(
            nn.Linear(dim, query_dim),
            nn.SiLU()
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, key_dim),
            nn.SiLU()
        )

        self.q_offset_scale = OffsetScale(query_dim, heads=2)
        self.k_offset_scale = OffsetScale(key_dim, heads=2)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
            self,
            q,
            k,
            v,
            mask=None,
            residual=True
    ):
        """
        b - batch
        tn - text sequence length (within groups)
        sn - speech sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, device, g = v.shape[0], v.device, self.group_size
        sn, tn = q.shape[-2], v.shape[-2]
        # prenorm

        normed_x = self.norm(v)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        q = self.to_q(q)
        k = self.to_k(k)

        # offset and scale
        quad_q, lin_q = self.q_offset_scale(q)
        quad_k, lin_k = self.k_offset_scale(k)

        # mask out linear attention keys

        if mask is not None:
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_mask = (lin_mask == 1)
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys

        if self.rotary_pos_emb:
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys,
                                               (quad_q, lin_q, quad_k, lin_k))

        # padding for groups

        quad_q, quad_k, lin_q, lin_k, v = map(
            lambda t: F.pad(t, (0, 0, 0, padding_to_multiple_of(t.shape[1], g)), value=0.),
            (quad_q, quad_k, lin_q, lin_k, v))

        mask = mask.bool() if mask is not None else torch.ones((b, tn), device=device, dtype=torch.bool)
        mask = F.pad(mask, (0, padding_to_multiple_of(mask.shape[1], g)), value=False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size),
                                              (quad_q, quad_k, lin_q, lin_k, v))

        if mask is not None:
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if mask is not None:
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / tn
            lin_out = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :q.shape[1]],
                                          (quad_out, lin_out))

        # gate

        out = gate * (quad_attn_out + lin_attn_out)

        out = self.to_out(out)
        # projection out and residual
        if residual:
            out = out + v

        quad_attn = rearrange(attn, 'b g n j -> b (g n) j')[:, :sn, :tn]

        return out, quad_attn
