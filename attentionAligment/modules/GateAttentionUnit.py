# -*- coding: utf-8 -*-
# @Time  : 2022/3/31 12:13
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : GateAttentionUnit.py
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        """
        :param x: input
        h： heads
        d： dim
        :return:
        """
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class GAU(nn.Module):
    """gated attention unit
    implementation of the gated attention unit from
    the paper Transformer Quality in Linear Time https://arxiv.org/pdf/2202.10447.pdf
    and  https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py
    """

    def __init__(
        self,
        *,
        dim,
        query_key_dim=128,
        expansion_factor=2.,
        add_residual=True,
        causal=False,
        dropout=0.,
        norm_klass=nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads=2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(
        self,
        x,
        rel_pos_bias=None,
        mask=None
    ):
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)

        z = self.to_qk(normed_x)
        q, k = self.offsetscale(z)

        qk = einsum('b i d, b j d -> b i j', q, k) / seq_len

        if rel_pos_bias is not None:
            qk = qk + rel_pos_bias

        attn = F.relu(qk) ** 2
        attn = self.dropout(attn)

        if mask is not None:
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out, attn


class GAUForTransformers(GAU):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        super().__init__(dim=self.hidden_size, dropout=config.hidden_dropout_prob)

