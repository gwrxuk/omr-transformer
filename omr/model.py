"""CNN + Transformer encoder-decoder for staff images -> music tokens.

    image 1x128xW
      -> CNN (height /16, width /8)          -> C x 8 x W/8
      -> flatten height into channels         -> W/8 frames
      -> linear + sinusoidal positions -> Transformer encoder
      -> Transformer decoder (causal, cross-attends to frames) -> token logits
"""
from __future__ import annotations

import math

import torch
from torch import nn


def conv_block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.MaxPool2d(pool),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B x T x d
        return x + self.pe[: x.shape[1]]


class OMRTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 enc_layers: int = 4, dec_layers: int = 4, ff: int = 1024,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.cnn = nn.Sequential(
            # strided stem: avoids full-resolution activations, the largest memory cost
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),  # 64 x W/2
            conv_block(32, 64, (2, 2)),    # 32 x W/4
            conv_block(64, 128, (2, 2)),   # 16 x W/8
            conv_block(128, 128, (2, 1)),  # 8  x W/8
        )
        self.proj = nn.Linear(128 * 8, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def encode(self, x: torch.Tensor, widths: torch.Tensor):
        f = self.cnn(x)                                  # B x C x 8 x T
        B, C, Hf, T = f.shape
        f = f.permute(0, 3, 1, 2).reshape(B, T, C * Hf)
        mem = self.pos(self.proj(f))
        frames = torch.div(widths + 7, 8, rounding_mode="floor").to(x.device)
        mem_pad = torch.arange(T, device=x.device)[None, :] >= frames[:, None]
        return self.encoder(mem, src_key_padding_mask=mem_pad), mem_pad

    def decode(self, tgt_in: torch.Tensor, mem: torch.Tensor, mem_pad: torch.Tensor):
        L = tgt_in.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.pos(self.embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(h, mem, tgt_mask=causal, tgt_key_padding_mask=tgt_in == self.pad_id,
                         memory_key_padding_mask=mem_pad)
        return self.head(self.norm(h))

    def forward(self, x, widths, tgt_in):
        mem, mem_pad = self.encode(x, widths)
        return self.decode(tgt_in, mem, mem_pad)

    @torch.no_grad()
    def greedy(self, x, widths, bos: int, eos: int, max_len: int = 128) -> list[list[int]]:
        mem, mem_pad = self.encode(x, widths)
        ys = torch.full((x.shape[0], 1), bos, dtype=torch.long, device=x.device)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for _ in range(max_len):
            nxt = self.decode(ys, mem, mem_pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return ys[:, 1:].tolist()

    @torch.no_grad()
    def beam(self, x, widths, bos: int, eos: int, k: int = 4, max_len: int = 128) -> list[int]:
        """Beam search for a single image (batch of 1)."""
        mem, mem_pad = self.encode(x[:1], widths[:1])
        beams = [([bos], 0.0)]
        finished = []
        for _ in range(max_len):
            cand = []
            ys = torch.tensor([b[0] for b in beams], device=x.device)
            logp = self.decode(ys, mem.expand(len(beams), -1, -1),
                               mem_pad.expand(len(beams), -1))[:, -1].log_softmax(-1)
            top = logp.topk(k, -1)
            for bi, (seq, score) in enumerate(beams):
                for v, t in zip(top.values[bi].tolist(), top.indices[bi].tolist()):
                    cand.append((seq + [t], score + v))
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in cand:
                (finished if seq[-1] == eos else beams).append((seq, score))
                if len(beams) == k:
                    break
            if len(finished) >= k or not beams:
                break
        best = max(finished or beams, key=lambda c: c[1] / len(c[0]))
        return best[0][1:]
