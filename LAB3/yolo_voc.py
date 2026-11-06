"""LAB 3 — Exercise: implement the YOLO v1 network and its loss, and train it on Pascal VOC 2007.

Everything else is ready: dataset download, target encoding, training loop, decoding
with NMS, mAP evaluation and TensorBoard logging (see yolo_tools.ipynb for an
explanation of each component). Your job is to complete the two TODO sections below:
    1. the `YOLONet` class (the network)
    2. the terms of the loss in `YoloLoss.forward`
Before training, the script checks both: the output shape of the network, and that the
loss is ~0 for a perfect prediction.

Run it with:
    python yolo_voc.py                              # full training (a GPU is strongly recommended)
    python yolo_voc.py --subset 500 --epochs 2      # quick test on a few images
    python yolo_voc.py --download_only              # just download the dataset (~870 MB)
    ... --data_dir D:/data                          # choose where to download it (default ~/Documents/datasets/VOC)

and follow the training with TensorBoard, from another terminal:
    tensorboard --logdir runs
"""
import argparse
import os
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, ops, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes, make_grid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "datasets", "VOC")   # default download folder

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
S = 7                 # the image is divided into S x S cells
B = 2                 # each cell predicts B boxes
K = len(VOC_CLASSES)  # number of classes (20)
IMG_SIZE = 224        # input resolution: a ResNet on 224x224 images produces a 7x7 feature map = S x S

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# TODO: implement the YOLO v1 network
# =============================================================================
class YOLONet(nn.Module):
    """YOLO v1 network.

    Input:  a batch of RGB images, shape (batch, 3, 224, 224), normalized with the ImageNet mean/std.
    Output: a tensor of shape (batch, S, S, B * 5 + K) = (batch, 7, 7, 30) with RAW values (no activation
            at the end: sigmoid/softmax are applied by the loss and by the decoder).

    For each cell, the 30 values are ordered as:
        [C1, x1, y1, w1, h1,  C2, x2, y2, w2, h2,  p_1, ..., p_20]
         \\---- box 1 ----/   \\---- box 2 ----/   \\-- class scores --/
    """

    def __init__(self, S=S, B=B, K=K):
        super().__init__()
        self.S, self.B, self.K = S, B, K

        # TODO: define your layers here, following the architecture of the slides:
        #
        #   1. BACKBONE: a pretrained ResNet50 without its last two layers (avgpool and fc)
        #      (On a CPU, models.resnet18 is much faster: its output has 512 channels instead of 2048.)
        #
        #   2. A 1x1 CONVOLUTION to reduce the number of channels
        #
        #   3. FULLY CONNECTED head with a final linear with S * S * (B * 5 + K) outputs.
        #
        # Optional: freeze the backbone to train faster (only the head is trained)
        # Optional: add an average pooling at the end of the backbone.
        # Optional: add dropout in the head to reduce overfitting.
        # Optional: add batch normalization in the head.
        # Optional: add a non-linearity (ReLU) after the 1x1 conv and/or in the head.
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool and fc
        # Freeze the pretrained feature extractor so only the YOLO head is trained.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Force the feature map to the S x S grid expected by the fully connected head.
        self.pool = nn.AdaptiveAvgPool2d((S, S))
        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * S * S, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, S * S * (B * 5 + K)),
        )

    def forward(self, x):
        # backbone -> 1x1 conv -> fully connected head -> grid output
        x = self.backbone(x)
        x = self.pool(x)
        x = self.reduce(x)
        x = self.head(x)
        return x.view(x.size(0), self.S, self.S, self.B * 5 + self.K)


# =============================================================================
# TODO: compose the YOLO v1 loss
# =============================================================================
class YoloLoss(nn.Module):
    """The YOLO v1 sum-squared loss (see yolo_tools.ipynb §7 for the formula).

    pred:   raw network output, shape (N, S, S, B*5 + K)
    target: encoded ground truth, shape (N, S, S, 5 + K) = [obj, x, y, w, h, one-hot class]
    Returns the total loss (a scalar) and a dict with the four terms (logged separately in TensorBoard).
    """

    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        # ---- Everything you need is already prepared here (the same steps as in the notebook) ----
        boxes, class_probs = split_predictions(pred)     # boxes (N,S,S,B,5) = [C, x, y, w, h]; class_probs (N,S,S,K)
        obj = target[..., 0]                             # (N,S,S): 1 if the cell contains an object -> 1^obj_i

        # Responsible predictor: among the B boxes of a cell, the one with the highest IoU with the ground truth
        pred_xyxy = ops.box_convert(cell_to_image(boxes[..., 1:5]), "cxcywh", "xyxy")               # (N,S,S,B,4)
        true_xyxy = ops.box_convert(cell_to_image(target[..., 1:5].unsqueeze(-2)), "cxcywh", "xyxy")  # (N,S,S,1,4)
        iou = box_iou_elementwise(pred_xyxy, true_xyxy.expand_as(pred_xyxy))                         # (N,S,S,B)
        best = iou.argmax(dim=-1, keepdim=True)
        resp = torch.zeros_like(iou).scatter_(-1, best, 1.0) * obj.unsqueeze(-1)   # (N,S,S,B): 1^obj_ij
        noobj = 1 - resp                                                            # (N,S,S,B): 1^noobj_ij

        tgt = target[..., 1:5].unsqueeze(-2)             # (N,S,S,1,4) ground-truth [x, y, w, h], broadcasts over B

        # Useful slices (they all broadcast together, shapes in brackets):
        #   predicted confidence  boxes[..., 0]    (N,S,S,B)      target confidence: 1 for responsible boxes, else 0
        #   predicted x, y        boxes[..., 1:3]  (N,S,S,B,2)    target x, y       tgt[..., 0:2]  (N,S,S,1,2)
        #   predicted w, h        boxes[..., 3:5]  (N,S,S,B,2)    target w, h       tgt[..., 2:4]  (N,S,S,1,2)
        #   predicted classes     class_probs      (N,S,S,K)      target classes    target[..., 5:] (N,S,S,K)

        # TODO: compute the terms of the loss, summing over all cells, boxes and images.
        #   Remember: square the errors, sum the last dimension (e.g. over x and y) with .sum(-1),
        #   multiply by the right mask (resp, noobj or obj) and then sum everything with .sum().
        #
        #   xy_loss    = ...   centre
        #   wh_loss    = ...   size
        #   conf_obj   = ...   confidence of the responsible boxes, target 1
        #   conf_noobj = ...   confidence of all the other boxes, target 0
        #   cls        = ...   classes
        #
        # Then average over the batch and return.
        #
        # The script checks your loss before training: a perfect prediction must give (almost) 0.

        # Coordinate and size errors are paid only by the responsible box.
        xy_loss = (((boxes[..., 1:3] - tgt[..., 0:2]) ** 2).sum(-1) * resp).sum()
        wh_loss = (((torch.sqrt(boxes[..., 3:5].clamp_min(1e-7))
                 - torch.sqrt(tgt[..., 2:4].clamp_min(1e-7))) ** 2).sum(-1) * resp).sum()

        # Responsible boxes should have confidence 1; all other boxes should have 0.
        conf_obj = (((boxes[..., 0] - 1) ** 2) * resp).sum()
        conf_noobj = ((boxes[..., 0] ** 2) * noobj).sum()

        # Classification is computed only for cells containing an object.
        cls = (((class_probs - target[..., 5:]) ** 2).sum(-1) * obj).sum()

        batch_size = pred.shape[0]
        coord = self.lambda_coord * (xy_loss + wh_loss) / batch_size
        conf_obj = conf_obj / batch_size
        conf_noobj = self.lambda_noobj * conf_noobj / batch_size
        cls = cls / batch_size
        total = coord + conf_obj + conf_noobj + cls
        return total, {"coord": coord, "conf_obj": conf_obj,
                   "conf_noobj": conf_noobj, "class": cls}


# =============================================================================
# From here on everything is ready: you don't need to modify anything
# =============================================================================

# ----------------------------------------------------------------------------- dataset
VOC_FILES = {
    "VOCtrainval_06-Nov-2007.tar": "trainval.txt",
    "VOCtest_06-Nov-2007.tar": "test.txt",
}
VOC_MIRRORS = [
    "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/",
    "https://pjreddie.com/media/files/",
]


def download_voc(root=DATA_DIR):
    """Download and extract Pascal VOC 2007 (trainval + test) into root/VOCdevkit/VOC2007."""
    voc_dir = os.path.join(root, "VOCdevkit", "VOC2007")
    os.makedirs(root, exist_ok=True)
    for tar_name, split_file in VOC_FILES.items():
        if os.path.exists(os.path.join(voc_dir, "ImageSets", "Main", split_file)):
            continue
        tar_path = os.path.join(root, tar_name)
        for mirror in VOC_MIRRORS:
            try:
                print(f"Downloading {mirror + tar_name} ...")
                request = urllib.request.Request(mirror + tar_name, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=60) as response, open(tar_path, "wb") as f:
                    total = int(response.headers.get("Content-Length", 0))
                    done, last_pct = 0, -10
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        pct = 100 * done // max(total, 1)
                        if pct >= last_pct + 10:
                            last_pct = pct - pct % 10
                            print(f"  {last_pct}%  ({done >> 20} / {total >> 20} MB)")
                break
            except Exception as e:
                print(f"  failed ({e}), trying the next mirror")
        else:
            raise RuntimeError(f"Could not download {tar_name} from any mirror")
        print(f"Extracting {tar_name} ...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(root)
        os.remove(tar_path)
    return voc_dir


def parse_annotation(xml_path):
    """Read a VOC xml file. Returns boxes (n, 4) as [xmin, ymin, xmax, ymax] in pixels, and labels (n,)."""
    root = ET.parse(xml_path).getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        if int(obj.find("difficult").text):          # skip the objects marked as "difficult"
            continue
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(VOC_CLASSES.index(obj.find("name").text))
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64)


def encode_target(boxes, labels):
    """Build the YOLO target for one image.

    boxes: (n, 4) normalized [cx, cy, w, h] in [0, 1]; labels: (n,)
    Returns a tensor (S, S, 5 + K): for the cell containing the centre of each object
    [1, x_cell, y_cell, w, h, one-hot class], where x_cell, y_cell are relative to the cell.
    """
    target = torch.zeros(S, S, 5 + K)
    for (cx, cy, w, h), label in zip(boxes.tolist(), labels.tolist()):
        col = min(int(cx * S), S - 1)
        row = min(int(cy * S), S - 1)
        if target[row, col, 0] == 1:        # YOLO v1: at most one object per cell
            continue
        target[row, col, :5] = torch.tensor([1.0, cx * S - col, cy * S - row, w, h])
        target[row, col, 5 + label] = 1.0
    return target


class VOCYolo(Dataset):
    def __init__(self, voc_dir, image_set, train):
        with open(os.path.join(voc_dir, "ImageSets", "Main", f"{image_set}.txt")) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.voc_dir = voc_dir
        self.train = train
        self.annotations = [parse_annotation(os.path.join(voc_dir, "Annotations", f"{i}.xml")) for i in self.ids]
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.ids)

    def load(self, idx):
        """Image resized to IMG_SIZE (not normalized) and normalized [cx, cy, w, h] boxes."""
        image = Image.open(os.path.join(self.voc_dir, "JPEGImages", f"{self.ids[idx]}.jpg")).convert("RGB")
        boxes, labels = self.annotations[idx]
        w, h = image.size
        boxes = boxes / torch.tensor([w, h, w, h])                       # xyxy in [0, 1]
        image = TF.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, ops.box_convert(boxes, "xyxy", "cxcywh"), labels

    def __getitem__(self, idx):
        image, boxes, labels = self.load(idx)
        if self.train:                                                   # data augmentation
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                boxes[:, 0] = 1 - boxes[:, 0]
            image = self.color(image)
        return self.normalize(TF.to_tensor(image)), encode_target(boxes, labels)

    def ground_truth(self, idx):
        """Boxes as normalized [xmin, ymin, xmax, ymax] and labels, for the evaluation."""
        boxes, labels = self.annotations[idx]
        with Image.open(os.path.join(self.voc_dir, "JPEGImages", f"{self.ids[idx]}.jpg")) as img:
            w, h = img.size
        return boxes / torch.tensor([w, h, w, h]), labels


# ----------------------------------------------------------------------------- loss and decoding
def split_predictions(pred):
    """Raw network output (N, S, S, B*5+K) -> boxes (N, S, S, B, 5) [C, x, y, w, h] and class probs (N, S, S, K).

    Sigmoid keeps confidence, centre and size in [0, 1]; softmax turns the class scores into probabilities.
    """
    boxes = torch.sigmoid(pred[..., : B * 5].reshape(*pred.shape[:3], B, 5))
    class_probs = torch.softmax(pred[..., B * 5 :], dim=-1)
    return boxes, class_probs


def cell_to_image(xywh):
    """(N, S, S, n_boxes, 4) with x, y relative to the cell -> [cx, cy, w, h] relative to the whole image."""
    rows = torch.arange(S, device=xywh.device).view(S, 1, 1)    # broadcast over (S, S, n_boxes)
    cols = torch.arange(S, device=xywh.device).view(1, S, 1)
    cx = (xywh[..., 0] + cols) / S
    cy = (xywh[..., 1] + rows) / S
    return torch.stack([cx, cy, xywh[..., 2], xywh[..., 3]], dim=-1)


def box_iou_elementwise(a, b, eps=1e-7):
    """IoU between corresponding boxes of a and b, both (..., 4) in [x1, y1, x2, y2]."""
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + eps)


def decode(pred, conf_thresh=0.2, nms_iou=0.5):
    """Raw output (N, S, S, B*5+K) -> for each image (boxes xyxy normalized, scores, labels) after NMS."""
    boxes, class_probs = split_predictions(pred)
    xyxy = ops.box_convert(cell_to_image(boxes[..., 1:5]), "cxcywh", "xyxy").clamp(0, 1)   # (N,S,S,B,4)
    cls_prob, cls_idx = class_probs.max(dim=-1)                                            # (N,S,S)
    scores = boxes[..., 0] * cls_prob.unsqueeze(-1)            # class-specific confidence C * P(class)
    labels = cls_idx.unsqueeze(-1).expand_as(scores)

    results = []
    for i in range(pred.shape[0]):
        b, s, l = xyxy[i].reshape(-1, 4), scores[i].reshape(-1), labels[i].reshape(-1)
        keep = s > conf_thresh
        b, s, l = b[keep], s[keep], l[keep]
        keep = ops.batched_nms(b, s, l, nms_iou)               # NMS separately for each class
        results.append((b[keep], s[keep], l[keep]))
    return results


# ----------------------------------------------------------------------------- evaluation
def average_precision(recall, precision):
    """VOC-style AP: area under the precision-recall curve (all-point interpolation)."""
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = torch.max(mpre[i], mpre[i + 1])
    idx = torch.nonzero(mrec[1:] != mrec[:-1]).squeeze(1)
    return ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()


def mean_average_precision(detections, ground_truths, iou_thresh=0.5):
    """detections / ground_truths: lists (one per image) of (boxes, scores, labels) / (boxes, labels).

    Returns the mAP and a dict {class index: AP} (only for the classes present in the ground truth).
    """
    aps = {}
    for c in range(K):
        gts = {i: g[0][g[1] == c] for i, g in enumerate(ground_truths)}
        n_gt = sum(len(g) for g in gts.values())
        if n_gt == 0:
            continue
        dets = [(i, s.item(), b) for i, (bs, ss, ls) in enumerate(detections) for b, s in zip(bs[ls == c], ss[ls == c])]
        dets.sort(key=lambda d: -d[1])
        matched = {i: torch.zeros(len(g), dtype=torch.bool) for i, g in gts.items()}
        tp = torch.zeros(len(dets))
        for k, (i, _, box) in enumerate(dets):
            if len(gts[i]) == 0:
                continue
            ious = ops.box_iou(box.unsqueeze(0), gts[i])[0]
            j = ious.argmax()
            if ious[j] >= iou_thresh and not matched[i][j]:
                tp[k] = 1
                matched[i][j] = True
        tp_cum = tp.cumsum(0)
        recall = tp_cum / n_gt
        precision = tp_cum / torch.arange(1, len(dets) + 1)
        aps[c] = average_precision(recall, precision) if len(dets) else 0.0
    return sum(aps.values()) / len(aps), aps


# ----------------------------------------------------------------------------- visualization
def denormalize(images):
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return ((images * std + mean).clamp(0, 1) * 255).to(torch.uint8)


def draw_detections(images, detections):
    """Images (N, 3, H, W) normalized + detections -> uint8 grid with boxes drawn, for TensorBoard."""
    drawn = []
    for img, (boxes, scores, labels) in zip(denormalize(images).cpu(), detections):
        names = [f"{VOC_CLASSES[l]} {s:.2f}" for l, s in zip(labels.tolist(), scores.tolist())]
        drawn.append(draw_bounding_boxes(img, boxes.cpu() * IMG_SIZE, labels=names, colors="red", width=2))
    return make_grid(drawn, nrow=4)


# ----------------------------------------------------------------------------- training
def check_model(net, device):
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=device))
    expected = (2, S, S, B * 5 + K)
    if tuple(out.shape) != expected:
        raise ValueError(f"The network must return a tensor of shape (batch, {S}, {S}, {B * 5 + K}), "
                         f"got {tuple(out.shape)}")
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Output shape OK: {tuple(out.shape)}")
    print(f"Parameters: {total:,} total, {trainable:,} trainable\n")


def perfect_prediction(target):
    """The raw output a perfect network would produce for the given targets (N, S, S, 5+K)."""
    pred = torch.full((*target.shape[:3], B * 5 + K), -12.0)          # sigmoid(-12) ~ 0: no object
    obj = target[..., 0] == 1
    pred[..., 0:5][obj] = torch.logit(target[..., 0:5][obj].clamp(1e-5, 1 - 1e-5))   # box 1 = ground truth
    pred[..., B * 5:][obj] = target[..., 5:][obj] * 30                 # softmax ~ one-hot
    return pred


def check_loss(loss_fn, dataset):
    """A perfect prediction must give a loss ~0, a random one a positive loss with gradients."""
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    target = torch.stack([base[i][1] for i in range(8)])

    loss_perfect, parts = loss_fn(perfect_prediction(target), target)
    if set(parts) != {"coord", "conf_obj", "conf_noobj", "class"}:
        raise ValueError(f"YoloLoss must return (total, dict with keys coord, conf_obj, conf_noobj, class), "
                         f"got keys {sorted(parts)}")
    random_pred = torch.randn(*target.shape[:3], B * 5 + K, requires_grad=True)
    loss_random, _ = loss_fn(random_pred, target)
    loss_random.backward()

    print(f"Loss check: perfect prediction {loss_perfect.item():.5f} (should be ~0), "
          f"random prediction {loss_random.item():.3f} (should be > 0)")
    if loss_random.dim() != 0 or not torch.isfinite(loss_random):
        raise ValueError("The loss must be a finite scalar")
    if loss_perfect.item() > 1e-3:
        raise ValueError("The loss of a perfect prediction is not ~0: check the terms of YoloLoss "
                         f"({ {k: round(v.item(), 4) for k, v in parts.items()} })")
    print()


def evaluate(net, loader, dataset, loss_fn, device, conf_thresh):
    net.eval()
    total_loss, detections = 0.0, []
    with torch.no_grad():
        for images, targets in loader:
            pred = net(images.to(device))
            total_loss += loss_fn(pred, targets.to(device))[0].item()
            detections += [tuple(t.cpu() for t in d) for d in decode(pred, conf_thresh)]
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    ground_truths = [base.ground_truth(i) for i in indices]
    mAP, aps = mean_average_precision(detections, ground_truths)
    return total_loss / len(loader), mAP, aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--subset", type=int, default=0, help="use only N train and N/2 test images (quick tests)")
    parser.add_argument("--conf_thresh", type=float, default=0.2, help="score threshold for the detections")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="where to download Pascal VOC (~900 MB)")
    parser.add_argument("--download_only", action="store_true")
    args = parser.parse_args()

    voc_dir = download_voc(args.data_dir)
    if args.download_only:
        return

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    train_set = VOCYolo(voc_dir, "trainval", train=True)
    test_set = VOCYolo(voc_dir, "test", train=False)
    if args.subset:
        train_set = Subset(train_set, range(args.subset))
        test_set = Subset(test_set, range(args.subset // 2))
    print(f"Images: train {len(train_set)}, test {len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=device == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device == "cuda")

    net = YOLONet().to(device)
    check_model(net, device)

    loss_fn = YoloLoss()
    check_loss(loss_fn, test_set)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = time.strftime("yolo_%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(HERE, "runs", run_name))
    print(f"TensorBoard logs in runs/{run_name}  ->  run:  tensorboard --logdir runs\n")
    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images[:8]                          # always visualize the same test images
    writer.add_graph(net, fixed_images[:1].to(device))

    best_map, step = 0.0, 0
    ckpt_path = os.path.join(HERE, "yolo_best.pth")
    for epoch in range(args.epochs):
        net.train()
        start = time.time()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            loss, parts = loss_fn(net(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("train/loss", loss.item(), step)
            for name, value in parts.items():
                writer.add_scalar(f"train_loss_parts/{name}", value.item(), step)
            step += 1
        scheduler.step()

        test_loss, mAP, aps = evaluate(net, test_loader, test_set, loss_fn, device, args.conf_thresh)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/mAP@0.5", mAP, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        net.eval()
        with torch.no_grad():
            dets = decode(net(fixed_images.to(device)), args.conf_thresh)
        writer.add_image("test/detections", draw_detections(fixed_images, dets), epoch)

        print(f"Epoch [{epoch + 1}/{args.epochs}]  train loss: {running / len(train_loader):.3f}  "
              f"test loss: {test_loss:.3f}  test mAP@0.5: {100 * mAP:.1f}%  ({time.time() - start:.0f}s)")
        if mAP > best_map:
            best_map = mAP
            torch.save(net.state_dict(), ckpt_path)

    print("\nAP per class (last epoch):")
    for c, ap in sorted(aps.items(), key=lambda t: -t[1]):
        print(f"  {VOC_CLASSES[c]:12s} {100 * ap:5.1f}%")
    print(f"\nBest test mAP@0.5: {100 * best_map:.1f}%  (model saved to {ckpt_path})")
    writer.close()


if __name__ == "__main__":
    main()
