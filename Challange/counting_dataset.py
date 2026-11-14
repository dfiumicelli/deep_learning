"""
PyTorch dataset and data loaders for the object counting dataset.

Layout (see the dataset README):
    <root>/<split>/<variant>/images/NNNNNN.png      RGB, 512 x 512
    <root>/<split>/<variant>/masks/NNNNNN.png       R = class id, G = instance id (0 = background)
    <root>/<split>/<variant>/counting.json          count and counts per class of every image
    <root>/<split>/<variant>/detection.json         COCO detection annotations
    <root>/test/images/NNNNNN.png                   test images, both variants mixed, no labels
split = train | val | test, variant = shadows | no_shadows. For the test split the variants are ignored and
targets['file_name'] is the name to use in the submission (e.g. 000123.png).

The challenge is scored on the count only, but any label can be used for training: choose them with `targets`,
any combination of 'count', 'boxes', 'masks' (or 'all'):

    loader = make_dataloader('dataset', 'train', targets=['count'], batch_size=32)
    for images, targets in loader:      # images: float [B, 3, H, W] in [0, 1]
        targets['count']                # int64 [B], number of objects
        targets['counts']               # int64 [B, C], objects per class (order of dataset.classes)

    loader = make_dataloader('dataset', 'train', targets=['count', 'boxes', 'masks'], batch_size=8)
    for images, targets in loader:      # the objects differ from image to image: one list entry per image
        targets['boxes'][i]             # float [N_i, 4], x1, y1, x2, y2 in pixels
        targets['labels'][i]            # int64 [N_i], class ids 1..C (0 is the background)
        targets['masks'][i]             # bool [N_i, H, W], one mask per object (same order as boxes / labels)
        split_targets(targets)          # list of B dicts, the format of the torchvision detectors

Fields per target:
    always                  image_id (int64 [B]), file_name (list of B strings)
    'count'                 count (int64 [B]), counts (int64 [B, C])
    'boxes'                 boxes, labels, area (visible pixels), iscrowd: lists of B tensors
    'masks'                 masks, labels, iscrowd: lists of B tensors
Only the counted objects (enough visible pixels) are in boxes / masks, unless include_ignored=True (then iscrowd marks
the others). Without the annotation files (e.g. a test set with hidden labels) the dataset is unlabelled: the targets
contain only image_id and file_name.

    python counting_dataset.py --root dataset --split val --targets all   # statistics and a preview grid
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VARIANTS = ('shadows', 'no_shadows')
TARGETS = ('count', 'boxes', 'masks')
STACKED = ('image_id', 'count', 'counts')  # same shape for every image: stacked in a batch; the rest become lists


def parse_targets(targets):
    """'count', 'all' or an iterable of TARGETS -> tuple in canonical order"""
    if isinstance(targets, str):
        targets = TARGETS if targets == 'all' else (targets,)
    targets = set(targets)
    unknown = targets - set(TARGETS)
    if unknown or not targets:
        raise ValueError(f"targets must be a non-empty combination of {TARGETS} (or 'all'), got {sorted(targets)}")
    return tuple(t for t in TARGETS if t in targets)


class CountingDataset(Dataset):
    """
    :param root: dataset folder (containing train, val, test)
    :param split: 'train', 'val' or 'test'
    :param variants: 'shadows', 'no_shadows' or both (a list): the images of the variants are concatenated
    :param targets: labels to load, any combination of 'count', 'boxes', 'masks', or 'all'
    :param transform: callable (image, target) -> (image, target); image is float [3, H, W] in [0, 1]. See Compose,
        RandomHorizontalFlip, Normalize
    :param include_ignored: boxes / masks: also the objects too hidden to be counted (with iscrowd = 1)
    """

    def __init__(self, root, split, variants=VARIANTS, targets='count', transform=None, include_ignored=False):
        self.root = Path(root)
        self.split = split
        self.variants = (variants,) if isinstance(variants, str) else tuple(variants)
        self.targets = parse_targets(targets)
        self.transform = transform
        self.include_ignored = include_ignored
        self.classes = None  # class names, index i = class id i + 1
        self.items = []
        self.labelled = True
        if (self.root / split / 'images').is_dir():
            # test split: one images/ folder, both variants mixed, no labels; file_name = the name to submit
            self.variants = ()
            self.labelled = False
            for path in sorted((self.root / split / 'images').glob('*.png')):
                self.items.append({'path': path, 'variant': None, 'image_id': int(path.stem), 'file_name': path.name})
        for variant in self.variants:
            folder = self.root / split / variant
            if not folder.is_dir():
                raise FileNotFoundError(f'{folder} not found')
            self._load_variant(folder, variant)
        if not self.items:
            raise RuntimeError(f'no images in {self.root / split} {self.variants}')

    # ---- loading ----

    def _load_variant(self, folder, variant):
        need_objects = 'boxes' in self.targets or 'masks' in self.targets
        counting_file, detection_file = folder / 'counting.json', folder / 'detection.json'
        if not counting_file.exists() or (need_objects and not detection_file.exists()):
            # unlabelled split: the images only
            self.labelled = False
            for path in sorted((folder / 'images').glob('*.png')):
                self.items.append({'path': path, 'variant': variant, 'image_id': int(path.stem),
                                   'file_name': f'{variant}/images/{path.name}'})
            return
        counting = json.loads(counting_file.read_text())
        self._check_classes(counting, counting_file)
        objects = {}
        if need_objects:
            detection = json.loads(detection_file.read_text())
            self._check_classes(detection, detection_file)
            for annotation in detection['annotations']:
                if annotation['counted'] or self.include_ignored:
                    objects.setdefault(annotation['image_id'], []).append(annotation)
        for image in counting['images']:
            self.items.append({
                'path': folder / image['file_name'], 'mask_path': folder / 'masks' / Path(image['file_name']).name,
                'variant': variant, 'image_id': image['id'], 'file_name': f"{variant}/{image['file_name']}",
                'count': image['count'], 'counts': [image['counts'][name] for name in self.classes],
                'annotations': objects.get(image['id'], []),
            })

    def _check_classes(self, data, path):
        classes = [c['name'] for c in sorted(data['categories'], key=lambda c: c['id'])]
        if self.classes is None:
            self.classes = classes
        elif classes != self.classes:
            raise ValueError(f'{path}: classes {classes} differ from {self.classes}')

    # ---- access ----

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = torch.from_numpy(np.array(Image.open(item['path']).convert('RGB'))).permute(2, 0, 1).float() / 255.0
        target = {'image_id': torch.tensor(item['image_id']), 'file_name': item['file_name']}
        if self.labelled:
            if 'count' in self.targets:
                target['count'] = torch.tensor(item['count'], dtype=torch.int64)
                target['counts'] = torch.tensor(item['counts'], dtype=torch.int64)
            if 'boxes' in self.targets or 'masks' in self.targets:
                target.update(self._object_targets(item, image.shape[1:]))
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def _object_targets(self, item, size):
        annotations = item['annotations']
        target = {
            'labels': torch.tensor([a['category_id'] for a in annotations], dtype=torch.int64),
            'iscrowd': torch.tensor([a['iscrowd'] for a in annotations], dtype=torch.int64),
        }
        if 'boxes' in self.targets:
            # x1, y1, x2, y2: the box covers the pixels x1 .. x2 - 1
            target['boxes'] = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                             a['bbox'][1] + a['bbox'][3]] for a in annotations],
                                           dtype=torch.float32).reshape(-1, 4)
            target['area'] = torch.tensor([a['visible_pixels'] for a in annotations], dtype=torch.float32)
        if 'masks' in self.targets:
            mask = np.array(Image.open(item['mask_path']).convert('RGB'))
            masks = [(mask[..., 0] == a['category_id']) & (mask[..., 1] == a['instance_id']) for a in annotations]
            target['masks'] = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, *size), dtype=torch.bool)
        return target

    def class_histogram(self):
        """Total objects per class and images per count, for checking the balance of a split"""
        if not self.labelled:
            return None
        per_class = np.sum([item['counts'] for item in self.items], axis=0)
        per_count = np.bincount([item['count'] for item in self.items])
        return dict(zip(self.classes, per_class.tolist())), per_count.tolist()


# ---- transforms: (image, target) -> (image, target) ----

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flips the image and, when present, the boxes and the masks (the counts do not change)"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        image = image.flip(-1)
        target = dict(target)
        if 'boxes' in target:
            width = image.shape[-1]
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = width - target['boxes'][:, [2, 0]]
            target['boxes'] = boxes
        if 'masks' in target:
            target['masks'] = target['masks'].flip(-1)
        return image, target


class Normalize:
    """Per channel (image - mean) / std, e.g. ImageNet statistics for pretrained backbones"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image, target):
        return (image - self.mean) / self.std, target


# ---- loaders ----

def collate(batch):
    """Images stacked [B, 3, H, W]; targets: one dict, fixed-size fields stacked (image_id, count, counts), the
    others (boxes, labels, masks, area, iscrowd, file_name) as lists with one entry per image"""
    images = torch.stack([image for image, _ in batch])
    targets = {}
    for key in batch[0][1]:
        values = [target[key] for _, target in batch]
        targets[key] = torch.stack(values) if key in STACKED else values
    return images, targets


def split_targets(targets):
    """Batch targets -> list of one dict per image, e.g. for the torchvision detectors
    (model(images, split_targets(targets)) with boxes and labels)"""
    size = len(targets['file_name'])
    return [{key: value[i] for key, value in targets.items()} for i in range(size)]


def make_dataloader(root, split, targets='count', variants=VARIANTS, batch_size=16, shuffle=None, num_workers=4,
                    transform=None, include_ignored=False, **loader_args):
    """
    DataLoader of a split with the chosen labels: targets = 'count', 'boxes', 'masks', any list of them, or 'all'.
    shuffle defaults to True for train. Extra arguments go to the DataLoader
    """
    dataset = CountingDataset(root, split, variants=variants, targets=targets, transform=transform,
                              include_ignored=include_ignored)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train') if shuffle is None else shuffle,
                      num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, **loader_args)


# ---- check ----

def _preview(images, targets, classes, path):
    from PIL import ImageDraw
    colours = [(255, 70, 70), (70, 220, 70), (70, 150, 255), (255, 210, 40), (220, 90, 255), (60, 220, 220)]
    tiles = []
    for image, target in zip(images, split_targets(targets)):
        tile = image.permute(1, 2, 0).clamp(0, 1).numpy()
        if 'masks' in target and len(target['masks']):
            # masks: tint the objects with their class colour
            for mask, label in zip(target['masks'].numpy(), target['labels'].tolist()):
                tile[mask] = 0.5 * tile[mask] + 0.5 * np.array(colours[(label - 1) % len(colours)]) / 255
        tile = Image.fromarray((tile * 255).astype(np.uint8))
        draw = ImageDraw.Draw(tile)
        for box, label in zip(target.get('boxes', []), target.get('labels', [])):
            colour = colours[(int(label) - 1) % len(colours)]
            draw.rectangle([float(v) for v in box], outline=colour, width=2)
            draw.text((float(box[0]) + 2, float(box[1]) + 1), classes[int(label) - 1], fill=colour)
        if 'count' in target:
            draw.rectangle([0, 0, 80, 14], fill=(0, 0, 0))
            draw.text((3, 1), f"count = {int(target['count'])}", fill=(255, 255, 255))
        tiles.append(tile)
    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    width, height = tiles[0].size
    grid = Image.new('RGB', (columns * width, rows * height))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % columns) * width, (i // columns) * height))
    grid.save(path)


def main():
    parser = argparse.ArgumentParser(description='Statistics of a split and a preview grid of one batch')
    parser.add_argument('--root', default='dataset')
    parser.add_argument('--split', default='val')
    parser.add_argument('--targets', nargs='+', default=['all'], help="count boxes masks, or all")
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--preview', default='batch_preview.png')
    args = parser.parse_args()

    targets = 'all' if args.targets == ['all'] else args.targets
    loader = make_dataloader(args.root, args.split, targets=targets, variants=args.variants,
                             batch_size=args.batch_size, shuffle=True, num_workers=2, transform=RandomHorizontalFlip(0.5))
    dataset = loader.dataset
    print(f'{args.split} {dataset.variants}: {len(dataset)} images, targets {dataset.targets}, classes {dataset.classes}')
    histogram = dataset.class_histogram()
    if histogram:
        per_class, per_count = histogram
        print('objects per class:', per_class)
        print('images per count:', {count: n for count, n in enumerate(per_count) if n})
    images, batch_targets = next(iter(loader))
    print('batch images:', tuple(images.shape), images.dtype)
    for key, value in batch_targets.items():
        if isinstance(value, torch.Tensor):
            print(f'  {key:10s} tensor {tuple(value.shape)}')
        else:
            shapes = [tuple(v.shape) if isinstance(v, torch.Tensor) else v for v in value[:3]]
            print(f'  {key:10s} list of {len(value)}, e.g. {shapes}')
    _preview(images, batch_targets, dataset.classes, args.preview)
    print('preview:', args.preview)


if __name__ == '__main__':
    main()
