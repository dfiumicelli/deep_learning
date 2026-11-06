"""LAB 4 — Exercise: build the encoder and decoder blocks of a Transformer and train it to translate dates.

Task: read a date written in many different human formats and output it in ISO format, e.g.
    "friday, march 5, 2021"   ->  "2021-03-05"
    "the 5th of march, 2021"  ->  "2021-03-05"
    "05.03.2021"              ->  "2021-03-05"
It is a real sequence-to-sequence problem (the encoder reads a sequence of characters, the decoder writes
another sequence character by character), but small enough to train in a couple of minutes.

Everything is ready: data generation, vocabularies, masks, multi-head attention, feed-forward network,
positional encoding, the full model, training loop, greedy decoding and TensorBoard logging
(see transformer_tools.ipynb for an explanation of each component). Your job is to complete the TODOs:
    1. the function `scaled_dot_product_attention`
    2. the `forward` of the `EncoderBlock` class (its layers are already created in __init__)
    3. the `forward` of the `DecoderBlock` class (its layers are already created in __init__)
Before training, the script checks each of them.

Run it with:
    python transformer_dates.py                     # default: 10 epochs
    python transformer_dates.py --epochs 3          # quicker

and follow the training with TensorBoard, from another terminal:
    tensorboard --logdir runs
"""
import argparse
import datetime
import math
import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# TODO 1: scaled dot-product attention
# =============================================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

    q:    queries, shape (B, h, Lq, d)
    k:    keys,    shape (B, h, Lk, d)
    v:    values,  shape (B, h, Lk, d)
    mask: optional boolean tensor, True = position to IGNORE; it broadcasts to (B, h, Lq, Lk)
          (e.g. (B, 1, 1, Lk) for the padding mask, (B, 1, Lq, Lk) for the decoder mask)

    Returns:
        out:     (B, h, Lq, d)   the weighted sums of the values
        weights: (B, h, Lq, Lk)  the attention weights (each row sums to 1)
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out, weights


# =============================================================================
# Ready-made building blocks (you will use them in the TODOs below)
# =============================================================================
class MultiHeadAttention(nn.Module):
    """Multi-head attention (uses your scaled_dot_product_attention).

    forward(query, key, value, mask) with query (B, Lq, D), key and value (B, Lk, D) -> (B, Lq, D)
      - self-attention:  query = key = value = x
      - cross-attention: query from the decoder, key = value = output of the encoder
    The attention weights of the last call are stored in self.last_attn, shape (B, h, Lq, Lk).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.last_attn = None

    def split_heads(self, x):
        """(B, L, D) -> (B, h, L, d)"""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.W_q(query))
        k = self.split_heads(self.W_k(key))
        v = self.split_heads(self.W_v(value))
        out, weights = scaled_dot_product_attention(q, k, v, mask)       # (B, h, Lq, d)
        self.last_attn = weights.detach()
        B, _, Lq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)          # concatenate the heads -> (B, Lq, D)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Dropout -> Linear, applied to each token. (B, L, D) -> (B, L, D)"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


# =============================================================================
# TODO 2: the forward of the encoder block
# =============================================================================
class EncoderBlock(nn.Module):
    """One encoder block (as in "Attention is all you need"):

        x -> [ self-attention  -> dropout -> + x -> LayerNorm ]
          -> [ feed-forward    -> dropout -> + x -> LayerNorm ] -> output

    forward(x, src_mask): x (B, Ls, D), src_mask (B, 1, 1, Ls) True on padding -> (B, Ls, D)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # The layers are ready: you only have to use them in forward()
        self.self_attn = MultiHeadAttention(d_model, n_heads)    # call it as self.self_attn(query, key, value, mask)
        self.ffn = FeedForward(d_model, d_ff, dropout)           # call it as self.ffn(x)
        self.norm1 = nn.LayerNorm(d_model)                       # after the self-attention sub-layer
        self.norm2 = nn.LayerNorm(d_model)                       # after the feed-forward sub-layer
        self.dropout = nn.Dropout(dropout)                       # applied to the output of each sub-layer

    def forward(self, x, src_mask):
        # TODO: the two sub-layers, each one with the pattern  x = norm(x + dropout(sublayer(...))):
        #   1. self-attention: queries, keys and values all come from x; use src_mask to ignore the padding
        #   2. feed-forward on the result
        # and return x (same shape as the input: (B, Ls, D))
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# =============================================================================
# TODO 3: the forward of the decoder block
# =============================================================================
class DecoderBlock(nn.Module):
    """One decoder block:

        y -> [ masked self-attention (over y)           -> dropout -> + y -> LayerNorm ]
          -> [ cross-attention (queries from y,
                                keys/values from memory) -> dropout -> + y -> LayerNorm ]
          -> [ feed-forward                             -> dropout -> + y -> LayerNorm ] -> output

    forward(y, memory, tgt_mask, src_mask):
        y        (B, Lt, D)       the (embedded) target tokens produced so far
        memory   (B, Ls, D)       the output of the encoder
        tgt_mask (B, 1, Lt, Lt)   True = ignore (future tokens and padding of the target)
        src_mask (B, 1, 1, Ls)    True = ignore (padding of the source)
        -> (B, Lt, D)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # The layers are ready: you only have to use them in forward()
        self.self_attn = MultiHeadAttention(d_model, n_heads)    # masked self-attention over the target
        self.cross_attn = MultiHeadAttention(d_model, n_heads)   # attention over the encoder output (memory)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)                       # after the masked self-attention
        self.norm2 = nn.LayerNorm(d_model)                       # after the cross-attention
        self.norm3 = nn.LayerNorm(d_model)                       # after the feed-forward
        self.dropout = nn.Dropout(dropout)                       # applied to the output of each sub-layer

    def forward(self, y, memory, tgt_mask, src_mask):
        # TODO: the three sub-layers, each one with the pattern  y = norm(y + dropout(sublayer(...))):
        #   1. masked self-attention: queries, keys and values from y, with tgt_mask
        #   2. cross-attention: queries from y, keys and values from memory, with src_mask
        #   3. feed-forward on the result
        # and return y (same shape as the input: (B, Lt, D))
        y = self.norm1(y + self.dropout(self.self_attn(y, y, y, tgt_mask)))
        y = self.norm2(y + self.dropout(self.cross_attn(y, memory, memory, src_mask)))
        y = self.norm3(y + self.dropout(self.ffn(y)))
        return y


# =============================================================================
# From here on everything is ready: you don't need to modify anything
# =============================================================================

# ----------------------------------------------------------------------------- data
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def ordinal(n):
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Human-readable formats of the same date d
FORMATS = [
    lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}",                        # march 5, 2021
    lambda d: f"{d.day} {MONTHS[d.month - 1]} {d.year}",                         # 5 march 2021
    lambda d: f"{MONTHS[d.month - 1][:3]} {d.day} {d.year}",                     # mar 5 2021
    lambda d: f"{WEEKDAYS[d.weekday()]}, {MONTHS[d.month - 1]} {d.day}, {d.year}",  # friday, march 5, 2021
    lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",                             # 05.03.2021
    lambda d: f"the {ordinal(d.day)} of {MONTHS[d.month - 1]}, {d.year}",        # the 5th of march, 2021
    lambda d: f"{MONTHS[d.month - 1][:3]}. {ordinal(d.day)} {d.year}",           # mar. 5th 2021
]


def random_date_pair(rng):
    """A random date between 1950 and 2049, as (human-readable string, ISO string)."""
    start = datetime.date(1950, 1, 1)
    d = start + datetime.timedelta(days=rng.randrange(365 * 100 + 24))
    return rng.choice(FORMATS)(d), d.isoformat()


# Character-level vocabularies. Index 0, 1, 2 are the special tokens.
PAD, SOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<sos>", "<eos>"]
SRC_VOCAB = SPECIALS + list(" ,.0123456789abcdefghijklmnopqrstuvwxyz")
TGT_VOCAB = SPECIALS + list("-0123456789")
SRC_STOI = {c: i for i, c in enumerate(SRC_VOCAB)}
TGT_STOI = {c: i for i, c in enumerate(TGT_VOCAB)}


def encode_src(text):
    return torch.tensor([SRC_STOI[c] for c in text])


def encode_tgt(text):
    return torch.tensor([SOS] + [TGT_STOI[c] for c in text] + [EOS])


def decode_tgt(ids):
    out = []
    for i in ids:
        if i == EOS:
            break
        if i not in (PAD, SOS):
            out.append(TGT_VOCAB[i])
    return "".join(out)


class DateDataset(Dataset):
    def __init__(self, n, seed):
        rng = random.Random(seed)
        self.pairs = [random_date_pair(rng) for _ in range(n)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return encode_src(src), encode_tgt(tgt)


def collate(batch):
    """Pad the sequences of a batch to the same length with PAD."""
    srcs, tgts = zip(*batch)
    src = nn.utils.rnn.pad_sequence(list(srcs), batch_first=True, padding_value=PAD)
    tgt = nn.utils.rnn.pad_sequence(list(tgts), batch_first=True, padding_value=PAD)
    return src, tgt


# ----------------------------------------------------------------------------- masks (True = ignore)
def make_src_mask(src):
    """(B, Ls) token ids -> (B, 1, 1, Ls): True on the padding of the source."""
    return (src == PAD)[:, None, None, :]


def make_tgt_mask(tgt_in):
    """(B, Lt) token ids -> (B, 1, Lt, Lt): True on future positions (causal mask) and on target padding."""
    Lt = tgt_in.shape[1]
    causal = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool, device=tgt_in.device), diagonal=1)   # (Lt, Lt)
    return causal[None, None, :, :] | (tgt_in == PAD)[:, None, None, :]


# ----------------------------------------------------------------------------- model
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding: adds to each position a fixed vector of sines and cosines."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, D), moved to the device with the model

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(len(SRC_VOCAB), d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(len(TGT_VOCAB), d_model, padding_idx=PAD)
        # As in the paper, the embeddings are multiplied by sqrt(d_model) (see encode/decode): we initialize them with
        # std 1/sqrt(d_model), so that the result has std ~1, like the positional encoding. With the default init
        # (std 1) the embeddings would be ~sqrt(d_model) times larger and would hide the position information.
        for emb in (self.src_embed, self.tgt_embed):
            nn.init.normal_(emb.weight, std=d_model ** -0.5)
            with torch.no_grad():
                emb.weight[PAD].zero_()
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, len(TGT_VOCAB))

    def encode(self, src, src_mask):
        x = self.dropout(self.pos(self.src_embed(src) * math.sqrt(self.d_model)))
        for block in self.encoder:
            x = block(x, src_mask)
        return x                                               # memory: (B, Ls, D)

    def decode(self, tgt_in, memory, src_mask):
        y = self.dropout(self.pos(self.tgt_embed(tgt_in) * math.sqrt(self.d_model)))
        tgt_mask = make_tgt_mask(tgt_in)
        for block in self.decoder:
            y = block(y, memory, tgt_mask, src_mask)
        return self.out(y)                                     # logits: (B, Lt, |TGT_VOCAB|)

    def forward(self, src, tgt_in):
        src_mask = make_src_mask(src)
        return self.decode(tgt_in, self.encode(src, src_mask), src_mask)


@torch.no_grad()
def greedy_decode(model, src, max_len=12):
    """Generate the output one token at a time, always taking the most likely next token."""
    model.eval()
    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)                       # the encoder runs only once
    ys = torch.full((src.shape[0], 1), SOS, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        logits = model.decode(ys, memory, src_mask)            # (B, current length, V)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (ys == EOS).any(dim=1).all():                       # every sequence has finished
            break
    return ys


def translate(model, texts, device):
    src = nn.utils.rnn.pad_sequence([encode_src(t) for t in texts], batch_first=True, padding_value=PAD).to(device)
    return [decode_tgt(seq.tolist()) for seq in greedy_decode(model, src)]


# ----------------------------------------------------------------------------- checks of the TODOs
def check_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
    mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
    mask[1, ..., 3:] = True                                    # the last 2 keys of the 2nd sequence are padding
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    if out.shape != (2, 3, 4, 8) or weights.shape != (2, 3, 4, 5):
        raise ValueError(f"Wrong shapes: out {tuple(out.shape)} (expected (2, 3, 4, 8)), "
                         f"weights {tuple(weights.shape)} (expected (2, 3, 4, 5))")
    if not torch.allclose(weights.sum(-1), torch.ones(2, 3, 4), atol=1e-5):
        raise ValueError("Each row of the attention weights must sum to 1 (softmax over the last dimension)")
    if weights[1, ..., 3:].abs().max() > 1e-6:
        raise ValueError("The masked positions must get attention weight 0")
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)   # PyTorch: True = attend
    if not torch.allclose(out, expected, atol=1e-5):
        raise ValueError("The output differs from torch.nn.functional.scaled_dot_product_attention: "
                         "did you divide by sqrt(d)?")
    print("TODO 1 scaled_dot_product_attention: OK")


def check_encoder_block(d_model=32):
    torch.manual_seed(0)
    block = EncoderBlock(d_model, n_heads=4, d_ff=64, dropout=0.1).eval()
    x = torch.randn(2, 6, d_model)
    src = torch.ones(2, 6, dtype=torch.long)
    src[1, 4:] = PAD                                           # 2nd sequence: 4 tokens + 2 padding
    mask = make_src_mask(src)
    out = block(x, mask)
    if out.shape != x.shape:
        raise ValueError(f"EncoderBlock must keep the shape {tuple(x.shape)}, got {tuple(out.shape)}")
    x2 = x.clone()
    x2[1, 4:] = torch.randn(2, d_model)                        # change only the padding positions
    if not torch.allclose(block(x2, mask)[1, :4], out[1, :4], atol=1e-5):
        raise ValueError("EncoderBlock: the real tokens must not depend on the padding: did you pass src_mask "
                         "to the self-attention?")
    print("TODO 2 EncoderBlock: OK")


def check_decoder_block(d_model=32):
    torch.manual_seed(0)
    block = DecoderBlock(d_model, n_heads=4, d_ff=64, dropout=0.1).eval()
    if not isinstance(getattr(block, "cross_attn", None), MultiHeadAttention):
        raise ValueError("DecoderBlock must have an attribute self.cross_attn = MultiHeadAttention(...)")
    y, memory = torch.randn(2, 5, d_model), torch.randn(2, 6, d_model)
    tgt_mask = make_tgt_mask(torch.ones(2, 5, dtype=torch.long))
    src_mask = make_src_mask(torch.ones(2, 6, dtype=torch.long))
    out = block(y, memory, tgt_mask, src_mask)
    if out.shape != y.shape:
        raise ValueError(f"DecoderBlock must keep the shape {tuple(y.shape)}, got {tuple(out.shape)}")
    y2 = y.clone()
    y2[:, 3:] = torch.randn(2, 2, d_model)                     # change only the "future" tokens 3 and 4
    if not torch.allclose(block(y2, memory, tgt_mask, src_mask)[:, :3], out[:, :3], atol=1e-5):
        raise ValueError("DecoderBlock: position t must not see the future tokens: did you pass tgt_mask "
                         "to the masked self-attention?")
    if torch.allclose(block(y, torch.randn_like(memory), tgt_mask, src_mask), out, atol=1e-5):
        raise ValueError("DecoderBlock: the output does not depend on the encoder: did you use memory as "
                         "keys and values of the cross-attention?")
    print("TODO 3 DecoderBlock: OK\n")


# ----------------------------------------------------------------------------- training and evaluation
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, writer, step):
    model.train()
    total = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]              # teacher forcing: input shifted right
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()                                       # update the learning rate at every step
        total += loss.item()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        step += 1
    return total / len(loader), step


def evaluate(model, loader, device):
    """Exact-match accuracy (whole date correct) and character accuracy, with greedy decoding."""
    exact, chars, n_chars, n = 0, 0, 0, 0
    for src, tgt in loader:
        preds = greedy_decode(model, src.to(device)).cpu()
        for p, t in zip(preds.tolist(), tgt.tolist()):
            p, t = decode_tgt(p), decode_tgt(t)
            exact += p == t
            chars += sum(a == b for a, b in zip(p, t))
            n_chars += len(t)
            n += 1
    return exact / n, chars / n_chars


def plot_cross_attention(model, text, device):
    """Heatmap of the cross-attention of the last decoder block: which input characters each output character looks at."""
    pred = translate(model, [text], device)[0]
    src = encode_src(text).unsqueeze(0).to(device)
    tgt_in = encode_tgt(pred)[:-1].unsqueeze(0).to(device)     # <sos> + prediction
    model.eval()
    with torch.no_grad():
        model(src, tgt_in)
    attn = model.decoder[-1].cross_attn.last_attn[0].mean(0).cpu()   # average over the heads: (Lt, Ls)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(text)), list(text))
    ax.set_yticks(range(len(pred) + 1), list(pred) + ["<eos>"])       # row t = character predicted at step t
    ax.set_xlabel("input characters")
    ax.set_ylabel("output characters")
    ax.set_title(f"Cross-attention: '{text}' -> '{pred}'")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_train", type=int, default=40_000)
    parser.add_argument("--no_plot", action="store_true", help="skip the final attention plot")
    args = parser.parse_args()

    check_attention()
    check_encoder_block()
    check_decoder_block()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_set, test_set = DateDataset(args.n_train, seed=0), DateDataset(2_000, seed=1)
    print("Some training pairs:")
    for src, tgt in train_set.pairs[:5]:
        print(f"  {src!r:36s} -> {tgt!r}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False, collate_fn=collate)

    model = Seq2SeqTransformer(args.d_model, args.n_heads, args.d_ff, args.n_layers).to(device)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}\n")
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)            # the padding does not count in the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Learning rate schedule: a short warmup (10% of the steps) up to args.lr, then a slow decrease.
    # Transformers are sensitive to the learning rate at the beginning of the training: the warmup helps a lot.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=args.epochs * len(train_loader))

    run_name = time.strftime("transformer_%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(HERE, "runs", run_name))
    print(f"TensorBoard logs in runs/{run_name}  ->  run:  tensorboard --logdir runs\n")
    examples = ["march 5, 2021", "the 1st of january, 2000", "dec. 25th 1999", "31.12.2049"]

    step = 0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, step = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, writer, step)
        exact, char_acc = evaluate(model, test_loader, device)
        writer.add_scalar("test/exact_match", exact, epoch)
        writer.add_scalar("test/char_accuracy", char_acc, epoch)
        preds = translate(model, examples, device)
        writer.add_text("examples", "  \n".join(f"{e} -> {p}" for e, p in zip(examples, preds)), epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  train loss: {train_loss:.4f}  test exact match: "
              f"{100 * exact:.1f}%  char accuracy: {100 * char_acc:.1f}%  ({time.time() - start:.0f}s)")

    print("\nTry it:")
    tests = examples + ["friday, july 4, 1980", "the 22nd of october, 1965", "aug 9 2031"]
    for text, pred in zip(tests, translate(model, tests, device)):
        print(f"  {text!r:32s} -> {pred}")

    torch.save(model.state_dict(), os.path.join(HERE, "transformer_dates.pth"))
    fig = plot_cross_attention(model, "tuesday, march 5, 2024", device)
    writer.add_figure("cross_attention", fig, args.epochs)
    writer.close()
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
