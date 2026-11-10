"""LAB 5 — Exercise: build a Vision Transformer (ViT) and train it on CIFAR-10.

"An image is worth 16x16 words" (Dosovitskiy et al., 2020): cut the image into patches, treat each patch as a
token, and classify the sequence with a standard Transformer encoder, reading a special [CLS] token.

Everything is ready: CIFAR-10 download, data augmentation, the layers of the model (created in the __init__
methods), a small CNN for comparison, training loop, evaluation, attention maps and TensorBoard logging (see
vit_tools.ipynb for an explanation of each component). Your job is to complete the TODOs:
    1. `PatchEmbedding.forward`: from a batch of images to a sequence of patch embeddings
    2. `ViT.embed`:              prepend the [CLS] token and add the positional embeddings
    3. `ViT.forward`:            Transformer encoder -> [CLS] token -> classification head
Before training, the script checks each of them.

Run it with:
    python vit_cifar.py                        # train the ViT (a GPU is strongly recommended)
    python vit_cifar.py --model cnn            # train the small CNN with the same recipe, to compare
    python vit_cifar.py --data_dir D:/data     # choose where to download CIFAR-10 (~150 MB, default ~/Documents/datasets)

and follow the training with TensorBoard, from another terminal:
    tensorboard --logdir runs
"""
import argparse
import io
import math
import os
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "datasets")    # default download folder

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# =============================================================================
# TODO 1: the forward of the patch embedding
# =============================================================================
class PatchEmbedding(nn.Module):
    """Cut the image into non-overlapping patch_size x patch_size patches and project each one to a vector of size dim.

    A Conv2d with kernel_size = stride = patch_size does exactly this: each output position "sees" one patch, and its
    dim output channels are a linear projection of the patch (see vit_tools.ipynb §3).

    forward(x): images (B, C, H, W) -> patch embeddings (B, N, dim), with N = (H / patch_size) * (W / patch_size),
                the patches ordered row by row (left to right, top to bottom).
    """

    def __init__(self, in_channels=3, patch_size=4, dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)   # (B, dim, H/p, W/p)

    def forward(self, x):
        # TODO:
        #   1. apply self.proj                          -> (B, dim, H/p, W/p): a grid of patch vectors
        #   2. flatten the two spatial dimensions       -> (B, dim, N)          (hint: .flatten(2))
        #   3. swap the last two dimensions             -> (B, N, dim)          (hint: .transpose(1, 2))
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


# =============================================================================
# The Vision Transformer: the layers are ready in __init__, TODO 2 and 3 are its methods
# =============================================================================
class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, n_classes=10, dim=192, depth=6, heads=6, mlp_dim=384, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(3, patch_size, dim)
        self.n_patches = (img_size // patch_size) ** 2                        # N = 64 for 32x32 images, 4x4 patches

        # The [CLS] token: ONE learned vector of size dim, shared by all the images, shape (1, 1, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # The positional embeddings: one learned vector for each of the N + 1 positions, shape (1, N + 1, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)

        # The Transformer encoder, made of PyTorch's ready-made blocks (pre-norm, GELU, as in the ViT paper)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)   # (B, L, dim) -> (B, L, dim)

        self.norm = nn.LayerNorm(dim)                     # final normalization of the [CLS] vector
        self.head = nn.Linear(dim, n_classes)             # classification head

    # -------------------------------------------------------------------------
    # TODO 2: from images to the input sequence of the encoder
    # -------------------------------------------------------------------------
    def embed(self, images):
        """images (B, 3, H, W) -> tokens (B, N + 1, dim): [CLS] followed by the N patch embeddings, plus positions."""
        # TODO:
                #   1. patch embeddings with self.patch_embed                      -> (B, N, dim)
                #   2. the [CLS] token is (1, 1, dim): repeat it for every image    -> (B, 1, dim)
                #      (hint: self.cls_token.expand(B, -1, -1), with B = images.shape[0])
                #   3. put it IN FRONT of the patch embeddings                      -> (B, N + 1, dim)   (hint: torch.cat(..., dim=1))
                #   4. add the positional embeddings self.pos_embed (broadcasting over the batch)
                #   5. apply self.dropout and return
        patch_tokens = self.patch_embed(images)
        batch_size = images.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        tokens = tokens + self.pos_embed
        return self.dropout(tokens)

    # -------------------------------------------------------------------------
    # TODO 3: the full forward pass
    # -------------------------------------------------------------------------
    def forward(self, images):
        """images (B, 3, H, W) -> class scores (logits) (B, n_classes)."""
        # TODO:
        #   1. tokens = self.embed(images)                                  -> (B, N + 1, dim)
        #   2. pass them through self.encoder                               -> (B, N + 1, dim)
        #   3. take the output of the [CLS] token (position 0)              -> (B, dim)
        #   4. apply self.norm and then self.head                           -> (B, n_classes)
        tokens = self.embed(images)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])  


# =============================================================================
# From here on everything is ready: you don't need to modify anything
# =============================================================================

# ----------------------------------------------------------------------------- a CNN to compare with
class SmallCNN(nn.Module):
    """A small CNN with the blocks of Lab 2 (Conv -> BatchNorm -> ReLU -> MaxPool) and global average pooling."""

    def __init__(self, n_classes=10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                                 nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 64), block(64, 128), block(128, 256))   # 32 -> 16 -> 8 -> 4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------- data
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def _download_from_huggingface(split, folder):
    """Download one split from the Hugging Face mirror (much faster than the official server) as uint8 arrays."""
    # Keep pyarrow optional: if it is unavailable, the caller falls back to
    # torchvision's official CIFAR-10 download.
    import importlib
    pq = importlib.import_module("pyarrow.parquet")

    path = os.path.join(folder, f"{split}.parquet")
    print(f"Downloading CIFAR-10 {split} split from Hugging Face ...")
    request = urllib.request.Request(HF_URL.format(split=split), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    table = pq.read_table(path, columns=["img", "label"])
    images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table.column("img").to_pylist()])
    labels = np.array(table.column("label").to_pylist(), dtype=np.int64)
    os.remove(path)
    return images, labels


def download_cifar10(data_dir=DATA_DIR):
    """Download CIFAR-10 (only the first time) into data_dir/cifar10 as two .npz files: train.npz and test.npz."""
    folder = os.path.join(data_dir, "cifar10")
    os.makedirs(folder, exist_ok=True)
    for split in ("train", "test"):
        npz_path = os.path.join(folder, f"{split}.npz")
        if os.path.exists(npz_path):
            continue
        try:
            images, labels = _download_from_huggingface(split, folder)
        except Exception as e:
            print(f"  Hugging Face download failed ({e}): using the official server via torchvision (it can be slow)")
            ds = datasets.CIFAR10(root=folder, train=split == "train", download=True)
            images, labels = ds.data, np.array(ds.targets, dtype=np.int64)
        np.savez(npz_path, images=images, labels=labels)
        print(f"  saved {npz_path}: {images.shape[0]} images")
    return folder


class CIFAR10(Dataset):
    """CIFAR-10 from the .npz files: images (N, 32, 32, 3) uint8 and labels (N,)."""

    def __init__(self, folder, train, transform):
        data = np.load(os.path.join(folder, "train.npz" if train else "test.npz"))
        self.images, self.labels = data["images"], data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


def get_datasets(data_dir=DATA_DIR):
    folder = download_cifar10(data_dir)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),         # data augmentation: random shifts...
        transforms.RandomHorizontalFlip(),            # ...and mirror images
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return CIFAR10(folder, train=True, transform=train_tf), CIFAR10(folder, train=False, transform=test_tf)


def denormalize(images):
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


# ----------------------------------------------------------------------------- checks of the TODOs
def check_patch_embedding():
    torch.manual_seed(0)
    pe = PatchEmbedding(3, patch_size=4, dim=16).eval()
    x = torch.randn(2, 3, 32, 32)
    out = pe(x)
    if out.shape != (2, 64, 16):
        raise ValueError(f"PatchEmbedding must return (B, N, dim) = (2, 64, 16), got {tuple(out.shape)}")
    x2 = x.clone()
    x2[:, :, 4:8, 8:12] += 1.0                        # change only the patch in row 1, column 2 -> token 1 * 8 + 2 = 10
    changed = (pe(x2) - out).abs().sum(dim=(0, 2)) > 1e-6
    if changed.nonzero().flatten().tolist() != [10]:
        raise ValueError("PatchEmbedding: the patches must be ordered row by row (token = row * 8 + column). "
                         "Did you use flatten(2) and transpose(1, 2)?")
    print("TODO 1 PatchEmbedding.forward: OK")


def check_embed():
    torch.manual_seed(0)
    model = ViT(dim=16, depth=1, heads=2, mlp_dim=32).eval()
    images = torch.randn(3, 3, 32, 32)
    tokens = model.embed(images)
    if tokens.shape != (3, 65, 16):
        raise ValueError(f"ViT.embed must return (B, N + 1, dim) = (3, 65, 16), got {tuple(tokens.shape)}")
    expected_cls = (model.cls_token + model.pos_embed[:, :1]).expand(3, -1, -1)
    if not torch.allclose(tokens[:, :1], expected_cls, atol=1e-6):
        if torch.allclose(tokens[:, :1], model.cls_token.expand(3, -1, -1), atol=1e-6):
            raise ValueError("ViT.embed: the [CLS] token is in the right place, but the positional embeddings "
                             "self.pos_embed were not added")
        raise ValueError("ViT.embed: the first token must be the [CLS] token plus pos_embed[:, 0]. "
                         "Did you put the [CLS] token in front (torch.cat([cls, patches], dim=1))?")
    expected_patches = model.patch_embed(images) + model.pos_embed[:, 1:]
    if not torch.allclose(tokens[:, 1:], expected_patches, atol=1e-5):
        raise ValueError("ViT.embed: tokens 1..N must be the patch embeddings plus the positional embeddings")
    print("TODO 2 ViT.embed: OK")


def check_forward():
    torch.manual_seed(0)
    model = ViT(dim=16, depth=2, heads=2, mlp_dim=32)
    out = model(torch.randn(3, 3, 32, 32))
    if out.shape != (3, 10):
        raise ValueError(f"ViT.forward must return (B, n_classes) = (3, 10), got {tuple(out.shape)}")
    out.sum().backward()
    for name in ["cls_token", "pos_embed", "head.weight", "norm.weight", "encoder.layers.0.self_attn.in_proj_weight"]:
        param = dict(model.named_parameters())[name]
        if param.grad is None or param.grad.abs().sum() == 0:
            raise ValueError(f"ViT.forward: the parameter {name} does not receive any gradient: is it used?")
    model.zero_grad()
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    tokens = model.encoder(model.embed(images))
    if not torch.allclose(model(images), model.head(model.norm(tokens[:, 0])), atol=1e-5):
        raise ValueError("ViT.forward: the logits must come from the [CLS] token (position 0) after the encoder, "
                         "self.norm and self.head")
    print("TODO 3 ViT.forward: OK\n")


# ----------------------------------------------------------------------------- attention maps
@torch.no_grad()
def cls_attention_maps(model, images):
    """How much the [CLS] token attends to each patch in the LAST encoder layer: (B, grid, grid), averaged over heads."""
    model.eval()
    x = model.embed(images)
    layers = model.encoder.layers
    for layer in layers[:-1]:
        x = layer(x)
    last = layers[-1]
    h = last.norm1(x)                                  # pre-norm: the attention sees the normalized tokens
    _, weights = last.self_attn(h, h, h, need_weights=True, average_attn_weights=True)   # (B, N + 1, N + 1)
    cls_to_patches = weights[:, 0, 1:]                 # row of the [CLS] query, columns of the patches
    grid = int(math.sqrt(cls_to_patches.shape[-1]))
    return cls_to_patches.reshape(-1, grid, grid)


def plot_attention(model, images, labels, device, n=8):
    images, labels = images[:n].to(device), labels[:n]
    preds = model(images).argmax(1).cpu()
    maps = cls_attention_maps(model, images).cpu()
    imgs = denormalize(images).cpu()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for i in range(n):
        axes[0, i].imshow(imgs[i].permute(1, 2, 0))
        color = "tab:green" if preds[i] == labels[i] else "tab:red"
        axes[0, i].set_title(f"{CIFAR10_CLASSES[preds[i]]}\n(true: {CIFAR10_CLASSES[labels[i]]})", fontsize=8, color=color)
        axes[1, i].imshow(imgs[i].permute(1, 2, 0))
        heat = nn.functional.interpolate(maps[i][None, None], size=imgs.shape[-2:], mode="bilinear")[0, 0]
        axes[1, i].imshow(heat, cmap="jet", alpha=0.5)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.suptitle("Top: predictions. Bottom: where the [CLS] token looks in the last layer")
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_position_similarity(model, positions=((0, 0), (0, 7), (3, 3), (4, 5), (7, 0), (7, 7))):
    """Cosine similarity between the learned positional embedding of some patches and all the others.

    Nobody tells the model that the patches lie on a 2D grid: if after training each position is similar to its
    neighbours, the model has learned the 2D structure of the image by itself.
    """
    pos = model.pos_embed[0, 1:].cpu()                               # (N, dim), without the [CLS] position
    pos = nn.functional.normalize(pos, dim=-1)
    sim = pos @ pos.T                                                # (N, N) cosine similarities
    grid = int(math.sqrt(pos.shape[0]))
    fig, axes = plt.subplots(1, len(positions), figsize=(2.2 * len(positions), 2.6))
    for ax, (r, c) in zip(axes, positions):
        ax.imshow(sim[r * grid + c].reshape(grid, grid), cmap="viridis", vmin=-1, vmax=1)
        ax.plot(c, r, "r*", ms=10)
        ax.set_title(f"patch ({r}, {c})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Similarity of the learned positional embeddings (red star = reference patch)")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------- training and evaluation
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, writer, step):
    model.train()
    total, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        step += 1
    return total / n, correct / n, step


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += loss_fn(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], default="vit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="where to download CIFAR-10 (~170 MB)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=0, help="use only N training and N test images (quick tests)")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    train_set, test_set = get_datasets(args.data_dir)
    if args.download_only:
        return
    if args.subset:
        train_set, test_set = Subset(train_set, range(args.subset)), Subset(test_set, range(args.subset))

    if args.model == "vit":
        check_patch_embedding()
        check_embed()
        check_forward()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Images: train {len(train_set)}, test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=device == "cuda")
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False, num_workers=args.num_workers,
                             pin_memory=device == "cuda")

    if args.model == "vit":
        model = ViT(32, args.patch_size, 10, args.dim, args.depth, args.heads, args.mlp_dim, args.dropout)
    else:
        model = SmallCNN()
    model = model.to(device)
    print(f"Model: {args.model}, parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=args.epochs * len(train_loader))

    run_name = time.strftime(f"{args.model}_%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(HERE, "runs", run_name))
    print(f"TensorBoard logs in runs/{run_name}  ->  run:  tensorboard --logdir runs\n")
    fixed_images, fixed_labels = next(iter(test_loader))

    step, best_acc = 0, 0.0
    ckpt_path = os.path.join(HERE, f"{args.model}_cifar10.pth")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, step = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device,
                                                      writer, step)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "test": test_acc}, epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  train loss: {train_loss:.3f}  train acc: {100 * train_acc:.1f}%  "
              f"test loss: {test_loss:.3f}  test acc: {100 * test_acc:.1f}%  ({time.time() - start:.0f}s)")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest test accuracy: {100 * best_acc:.1f}%  (model saved to {ckpt_path})")
    if args.model == "vit":
        writer.add_figure("cls_attention", plot_attention(model, fixed_images, fixed_labels, device), args.epochs)
        writer.add_figure("position_similarity", plot_position_similarity(model), args.epochs)
        if not args.no_plot:
            plt.show()
    writer.close()


if __name__ == "__main__":
    main()
