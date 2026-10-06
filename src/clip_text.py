"""CLIP text tower (training time only): initialises attribute queries from their names.

Re-implements OpenAI's BPE tokenizer and text transformer on top of ``src.vit``,
because the released TorchScript archives hard-code CUDA. Query ``i`` is set to
``proj @ text(prompt_i)``: the visual projection maps image tokens into CLIP's
joint space, so its transpose maps text embeddings back into token space.
"""
from __future__ import annotations

import gzip
import re
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attributes import ATTRIBUTE_NAMES, GROUPS
from .model import NONE_GROUPS, CSARNet, QueryHead
from .pretrained import CLIP_RELEASES, _download, clip_state
from .vit import Block, ViTConfig

BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
SOT, EOT, CONTEXT = 49406, 49407, 77

_COLOURS = ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow")
PROMPTS: dict[str, str] = {
    "Age-Young": "a young person", "Age-Adult": "an adult person", "Age-Old": "an old person",
    "Gender-Female": "a woman",
    "Hair-Length-Short": "a person with short hair", "Hair-Length-Long": "a person with long hair",
    "Hair-Length-Bald": "a bald person",
    "UpperBody-Length-Short": "a person wearing short sleeves",
    **{f"UpperBody-Color-{c.capitalize()}": f"a person wearing a {c} top" for c in _COLOURS},
    "UpperBody-Color-Other": "a person wearing a top of an unusual colour",
    "LowerBody-Length-Short": "a person wearing shorts",
    **{f"LowerBody-Color-{c.capitalize()}": f"a person wearing {c} trousers" for c in _COLOURS},
    "LowerBody-Color-Other": "a person wearing trousers of an unusual colour",
    "LowerBody-Type-Trousers&Shorts": "a person wearing trousers or shorts",
    "LowerBody-Type-Skirt&Dress": "a person wearing a skirt or a dress",
    "Accessory-Backpack": "a person with a backpack", "Accessory-Bag": "a person carrying a bag",
    "Accessory-Glasses-Normal": "a person wearing glasses", "Accessory-Glasses-Sun": "a person wearing sunglasses",
    "Accessory-Hat": "a person wearing a hat",
}
NONE_PROMPTS = {"hair": "a person whose hair is hidden", "lower_type": "a person whose legs are hidden",
                "glasses": "a person without glasses"}


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    """OpenAI CLIP BPE for lower-case ASCII prompts (the regex keeps CLIP's split rules for ASCII)."""

    _split = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+")

    def __init__(self):
        merges = gzip.open(_download(BPE_URL, "clip_bpe_simple_vocab_16e6.txt.gz")).read().decode().split("\n")
        merges = [tuple(m.split()) for m in merges[1:49152 - 256 - 2 + 1]]
        self.byte_encoder = _bytes_to_unicode()
        vocab = list(self.byte_encoder.values())
        vocab += [v + "</w>" for v in vocab] + ["".join(m) for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}

    @lru_cache(maxsize=None)
    def _bpe(self, token: str) -> tuple[str, ...]:
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        return word

    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in self._split.findall(" ".join(text.lower().split())):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids += [self.encoder[t] for t in self._bpe(tok)]
        return ids

    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), CONTEXT, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = [SOT, *self.encode(text)[:CONTEXT - 2], EOT]
            out[i, :len(ids)] = torch.tensor(ids)
        return out


class TextEncoder(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor]):
        super().__init__()
        width = sd["ln_final.weight"].shape[0]
        depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.resblocks."))
        cfg = ViTConfig(1, width, depth, width // 64, 4 * width, (1, 1), act="quick_gelu", eps=1e-5)
        self.token_embedding = nn.Embedding.from_pretrained(sd["token_embedding.weight"].float())
        self.pos = nn.Parameter(sd["positional_embedding"].float())
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(depth))
        self.ln_final = nn.LayerNorm(width, eps=1e-5)
        self.proj = nn.Parameter(sd["text_projection"].float())
        state = {"ln_final.weight": sd["ln_final.weight"], "ln_final.bias": sd["ln_final.bias"]}
        for i in range(depth):
            s, b = f"transformer.resblocks.{i}.", f"blocks.{i}."
            state[f"{b}attn.qkv.weight"], state[f"{b}attn.qkv.bias"] = sd[f"{s}attn.in_proj_weight"], sd[f"{s}attn.in_proj_bias"]
            for src, dst in (("ln_1", "norm1"), ("ln_2", "norm2"), ("attn.out_proj", "attn.proj"),
                             ("mlp.c_fc", "fc1"), ("mlp.c_proj", "fc2")):
                state[f"{b}{dst}.weight"], state[f"{b}{dst}.bias"] = sd[f"{s}{src}.weight"], sd[f"{s}{src}.bias"]
        self.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.pos
        for blk in self.blocks:
            x = blk(x, causal=True)
        x = self.ln_final(x)
        return x[torch.arange(len(x)), tokens.argmax(-1)] @ self.proj


def query_prompts() -> list[str]:
    return [PROMPTS[n] for n in ATTRIBUTE_NAMES] + [NONE_PROMPTS[GROUPS[k].name] for k in NONE_GROUPS]


@torch.no_grad()
def init_queries_from_text(model: CSARNet) -> None:
    """Set the QueryHead queries of a CLIP-backbone model from CLIP text embeddings."""
    if model.cfg.backbone not in CLIP_RELEASES or not isinstance(model.head, QueryHead):
        raise ValueError("text initialisation needs head='query' and a clip_* backbone")
    sd = clip_state(*CLIP_RELEASES[model.cfg.backbone])
    text = TextEncoder(sd).eval()(Tokenizer()(query_prompts()))
    queries = F.normalize(text, dim=-1) @ sd["visual.proj"].float().T
    model.head.queries.copy_(F.normalize(queries, dim=-1) * model.head.queries.shape[-1] ** 0.5)
