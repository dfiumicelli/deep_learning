"""LAB 5 — Exercise (going further): fine-tune a ViT-B/16 pretrained on ImageNet on CIFAR-10.

In vit_cifar.py we trained a small ViT from scratch on 50,000 images. Here we start instead from torchvision's
ViT-B/16 (86M parameters, 16x16 patches on 224x224 images, pretrained on ImageNet, see vit_tools.ipynb §9) and adapt
it to CIFAR-10 (→ Lab 3, pretrained weights):
    1. the 32x32 CIFAR images are upscaled to 224x224 and normalized with the ImageNet statistics, exactly as the
       images seen during pretraining;
    2. the final head (768 -> 1000 ImageNet classes) is replaced by a new one (768 -> 10 CIFAR classes);
    3. the network is trained with a SMALL learning rate for the pretrained backbone (we only want to adjust it, not
       destroy what it has learned) and a larger one for the new head (which starts from scratch).

Two modes:
    --mode full     fine-tuning: all the 86M parameters are trained (≈ 98–99% test accuracy in 2–3 epochs)
    --mode linear   linear probing: the backbone is frozen, only the head is trained (7,690 parameters!): how good
                    are the ImageNet features, as they are, for CIFAR-10?

Everything is ready (data, training loop with mixed precision, evaluation, TensorBoard logging) except the model:
your job is to complete the TODO in `build_model`:
    load the pretrained ViT-B/16, (in linear mode) freeze it, and replace its last layer with a new nn.Linear.
Before training, the script checks it.

Run it with:
    python vit_finetune.py                              # full fine-tuning, 3 epochs (a GPU is needed: ~10 min/epoch on a laptop GPU)
    python vit_finetune.py --mode linear                # linear probing
    python vit_finetune.py --batch_size 32              # if you run out of GPU memory
    python vit_finetune.py --subset 1000 --epochs 1     # quick test of the pipeline

and follow the training with TensorBoard, from another terminal:
    tensorboard --logdir runs
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

from vit_cifar import CIFAR10, CIFAR10_CLASSES, DATA_DIR, download_cifar10

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- data
def get_datasets(data_dir, weights):
    """CIFAR-10 with the preprocessing of the pretrained model: resize to 224x224 + ImageNet normalization."""
    folder = download_cifar10(data_dir)
    pre = weights.transforms()                       # what the model expects: crop size, interpolation, mean and std
    size = pre.crop_size[0]                          # 224
    normalize = transforms.Normalize(pre.mean, pre.std)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),         # the same data augmentation as vit_cifar.py...
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size, interpolation=pre.interpolation),   # ...then 32x32 -> 224x224
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(size, interpolation=pre.interpolation),
        transforms.ToTensor(),
        normalize,
    ])
    return CIFAR10(folder, train=True, transform=train_tf), CIFAR10(folder, train=False, transform=test_tf)


def denormalize(images, weights):
    pre = weights.transforms()
    mean = torch.tensor(pre.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(pre.std, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


# =============================================================================
# TODO: build the model
# =============================================================================
def build_model(weights, n_classes=10, mode="full"):
    """ViT-B/16 pretrained on ImageNet, with a new classification head for n_classes classes.

    weights:  the pretrained weights to load (models.ViT_B_16_Weights.DEFAULT)
    mode:     "full"   -> all the parameters stay trainable
              "linear" -> the pretrained network is frozen, only the new head is trainable
    """
    # TODO:
    #   1. create the ViT-B/16 with the pretrained weights            (hint: models.vit_b_16(weights=...), see
    #                                                                  vit_tools.ipynb §9)
    #   2. only if mode == "linear": freeze ALL its parameters         (hint: loop over model.parameters() and set
    #                                                                  p.requires_grad = False)
    #   3. replace the last layer, model.heads, which maps the [CLS] vector (768 numbers) to the 1000 ImageNet
    #      classes, with a new nn.Linear from model.hidden_dim (= 768) to n_classes
    #      (try print(model) to see the layers: the new head is just an attribute assignment, model.heads = ...)
    #   4. return the model
    # Question: why must step 3 come AFTER step 2?
    raise NotImplementedError("TODO: implement build_model")


def new_head(model):
    """The last layer of the model: model.heads, or model.heads.head if only the Linear inside it was replaced."""
    if isinstance(model.heads, nn.Sequential) and len(model.heads) == 1:
        return model.heads[0]
    return model.heads


def check_model(model, weights, n_classes, mode):
    """Check the model built by build_model (on the CPU, before training)."""
    if not isinstance(model, models.VisionTransformer):
        raise ValueError(f"build_model must return the torchvision ViT-B/16 (models.vit_b_16), got {type(model)}")
    reference = weights.get_state_dict(progress=True)
    for name in ("conv_proj.weight", "encoder.pos_embedding"):
        if not torch.equal(model.state_dict()[name].cpu(), reference[name]):
            raise ValueError(f"The backbone does not have the pretrained weights ({name} differs from ImageNet's): "
                             "did you pass weights=weights to models.vit_b_16?")
    head = new_head(model)
    if not isinstance(head, nn.Linear):
        raise ValueError(f"model.heads must be a single nn.Linear, got:\n{model.heads}")
    if (head.in_features, head.out_features) != (model.hidden_dim, n_classes):
        raise ValueError(f"The new head must map {model.hidden_dim} -> {n_classes}, "
                         f"got {head.in_features} -> {head.out_features}")
    if not all(p.requires_grad for p in head.parameters()):
        raise ValueError("The new head is frozen: did you replace model.heads BEFORE freezing the parameters?")
    backbone = [p for name, p in model.named_parameters() if not name.startswith("heads")]
    if mode == "full" and not all(p.requires_grad for p in backbone):
        raise ValueError("In mode 'full' all the parameters must be trainable: freeze them only in mode 'linear'")
    if mode == "linear" and any(p.requires_grad for p in backbone):
        raise ValueError("In mode 'linear' the backbone must be frozen (requires_grad = False)")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    if out.shape != (2, n_classes):
        raise ValueError(f"The model must return (B, n_classes) = (2, {n_classes}), got {tuple(out.shape)}")
    print(f"TODO build_model ({mode}): OK")


def count(params):
    return sum(p.numel() for p in params)


# ----------------------------------------------------------------------------- training and evaluation
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, amp_dtype, device, writer, step, epoch):
    model.train()
    total, correct, n = 0.0, 0, 0
    start = time.time()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # Mixed precision: the matrix products run in 16 bit (≈ 2x faster, half the memory for the activations)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                 # with float16 the loss is scaled to avoid underflow
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr_backbone", scheduler.get_last_lr()[0], step)
        step += 1
        if (i + 1) % 100 == 0:
            print(f"  epoch {epoch + 1}  batch [{i + 1}/{len(loader)}]  loss: {total / n:.3f}  "
                  f"acc: {100 * correct / n:.1f}%  ({n / (time.time() - start):.0f} img/s)")
    return total / n, correct / n, step


@torch.no_grad()
def evaluate(model, loader, loss_fn, amp_dtype, device, n_classes=10):
    model.eval()
    total, n = 0.0, 0
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)    # rows: true class, columns: predicted class
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        total += loss_fn(logits.float(), labels).item() * len(labels)
        n += len(labels)
        idx = labels.cpu() * n_classes + logits.argmax(1).cpu()
        confusion += torch.bincount(idx, minlength=n_classes ** 2).view(n_classes, n_classes)
    return total / n, confusion.diag().sum().item() / n, confusion


@torch.no_grad()
def plot_predictions(model, images, labels, weights, amp_dtype, device, n=16):
    model.eval()
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
        probs = model(images[:n].to(device)).float().softmax(-1).cpu()
    shown = denormalize(images[:n], weights)
    fig, axes = plt.subplots(2, n // 2, figsize=(1.9 * n // 2, 4.6))
    for j, ax in enumerate(axes.flat):
        pred = probs[j].argmax().item()
        ax.imshow(shown[j].permute(1, 2, 0))
        ax.set_title(f"{CIFAR10_CLASSES[pred]} ({100 * probs[j, pred]:.0f}%)\ntrue: {CIFAR10_CLASSES[labels[j]]}",
                     fontsize=8, color="green" if pred == labels[j] else "red")
        ax.axis("off")
    fig.suptitle("Fine-tuned ViT-B/16 on CIFAR-10 test images (224x224 input)")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "linear"], default="full",
                        help="full: fine-tune all the parameters; linear: train only the new head")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate of the pretrained backbone")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="learning rate of the new head")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="where to download CIFAR-10 (~170 MB)")
    parser.add_argument("--ckpt", type=str, default=os.path.join(DATA_DIR, "checkpoints", "vit_b16_cifar10.pth"),
                        help="where to save the best model (~330 MB: outside OneDrive by default)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=0, help="use only N training and N test images (quick tests)")
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision (slower, more memory)")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    weights = models.ViT_B_16_Weights.DEFAULT           # IMAGENET1K_V1 (~330 MB download the first time)
    train_set, test_set = get_datasets(args.data_dir, weights)
    if args.subset:
        train_set, test_set = Subset(train_set, range(args.subset)), Subset(test_set, range(args.subset))

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found. Fine-tuning an 86M-parameter ViT on the CPU takes many hours: "
              "try --subset 1000 --epochs 1 to just see the pipeline running.")
    # bfloat16 if the GPU supports it (no loss scaling needed), otherwise float16 + GradScaler; no AMP on CPU/MPS
    amp_dtype = None
    if device == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device if device == "cuda" else "cpu", enabled=amp_dtype == torch.float16)
    print(f"Using device: {device}  (mixed precision: {amp_dtype})")
    print(f"Images: train {len(train_set)}, test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=device == "cuda",
                              persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=2 * args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=device == "cuda", persistent_workers=args.num_workers > 0)

    model = build_model(weights, n_classes=len(CIFAR10_CLASSES), mode=args.mode)
    check_model(model, weights, len(CIFAR10_CLASSES), args.mode)
    nn.init.zeros_(new_head(model).weight)            # the new head starts from "uniform" predictions (as in the
    nn.init.zeros_(new_head(model).bias)              # ViT paper), so it does not disturb the backbone at the start
    model = model.to(device)
    head_params = list(model.heads.parameters())
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("heads") and p.requires_grad]
    print(f"Model: ViT-B/16 ({args.mode}), parameters: {count(model.parameters()):,}, "
          f"trainable: {count(backbone_params) + count(head_params):,}\n")

    # Different learning rates for the backbone and the head. As usual for Transformers, weight decay only on the
    # weights of the Linear and Conv2d layers, not on biases, LayerNorms, [CLS] token and positional embeddings
    decay = [p for p in backbone_params if p.ndim == 2 or p.ndim == 4]       # Linear (2D) and Conv2d (4D) weights
    no_decay = [p for p in backbone_params if not (p.ndim == 2 or p.ndim == 4)]
    groups = [{"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}]
    if backbone_params:
        groups = [{"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
                  {"params": no_decay, "lr": args.lr, "weight_decay": 0.0}] + groups
    optimizer = torch.optim.AdamW(groups)
    # warmup + cosine decay, each group to its own max learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], pct_start=0.1,
                                                    total_steps=args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_name = time.strftime(f"vit_b16_{args.mode}_%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(HERE, "runs", run_name))
    print(f"TensorBoard logs in runs/{run_name}  ->  run:  tensorboard --logdir runs\n")
    fixed_images, fixed_labels = next(iter(test_loader))
    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)

    step, best_acc, best_confusion = 0, 0.0, None
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, step = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler,
                                                      amp_dtype, device, writer, step, epoch)
        test_loss, test_acc, confusion = evaluate(model, test_loader, loss_fn, amp_dtype, device)
        writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "test": test_acc}, epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  train loss: {train_loss:.3f}  train acc: {100 * train_acc:.1f}%  "
              f"test loss: {test_loss:.3f}  test acc: {100 * test_acc:.1f}%  ({time.time() - start:.0f}s)")
        if test_acc > best_acc:
            best_acc, best_confusion = test_acc, confusion
            torch.save(model.state_dict(), args.ckpt)

    print(f"\nBest test accuracy: {100 * best_acc:.2f}%  (model saved to {args.ckpt})")
    print("Per-class accuracy:")
    per_class = best_confusion.diag() / best_confusion.sum(1).clamp(min=1)
    for name, acc in zip(CIFAR10_CLASSES, per_class):
        print(f"  {name:10s} {100 * acc:5.1f}%")

    writer.add_figure("predictions", plot_predictions(model, fixed_images, fixed_labels, weights, amp_dtype, device),
                      args.epochs)
    writer.close()
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
