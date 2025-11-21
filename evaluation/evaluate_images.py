"""
Evaluate generated images using Mask2Former (or other object detector model)
"""

# Workaround for missing petrel_client (optional dependency for cloud storage backends)
# This must be done before importing mmdet/mmengine
import sys
from unittest.mock import MagicMock

# Mock petrel_client module before it's imported by mmengine
if 'petrel_client' not in sys.modules:
    sys.modules['petrel_client'] = MagicMock()
    # Also mock common submodules that might be accessed
    petrel_mock = MagicMock()
    sys.modules['petrel_client.petrel'] = petrel_mock
    sys.modules['petrel_client.client'] = petrel_mock

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

# Get directory path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./")
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            # "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
        )
    return args

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

# Load models

@timed
def load_models(args):
    CONFIG_PATH = args.model_config
    OBJECT_DETECTOR = args.options.get('model', "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco")
    CKPT_PATH = os.path.join(args.model_path, f"{OBJECT_DETECTOR}.pth")
    object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = args.options.get('clip_model', "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}

# Evaluation parts

class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        bgcolor = args.options.get('bgcolor', "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if args.options.get('crop', '1') == '1':
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (transform(image), 0)


def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes),
        batch_size=16, num_workers=4
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                # Color check
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                # Relative position check
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    
    # Handle new API: DetDataSample object
    if hasattr(result, 'pred_instances'):
        pred_instances = result.pred_instances
        # Convert to old format: list of arrays per class
        bboxes = pred_instances.bboxes.cpu().numpy()  # (N, 4)
        labels = pred_instances.labels.cpu().numpy()  # (N,)
        scores = pred_instances.scores.cpu().numpy()  # (N,)
        masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') and pred_instances.masks is not None else None
        
        # Convert to old format: bbox[index] is array of shape (M, 5) for class index
        num_classes = len(classnames)
        bbox = [np.zeros((0, 5)) for _ in range(num_classes)]
        segm = None
        segm_list = None
        
        if masks is not None:
            segm_list = [[] for _ in range(num_classes)]
        
        for i in range(len(labels)):
            cls_idx = int(labels[i])
            if 0 <= cls_idx < num_classes:
                # Append [x1, y1, x2, y2, score]
                bbox_arr = np.concatenate([bboxes[i], scores[i:i+1]]).reshape(1, -1)
                bbox[cls_idx] = np.vstack([bbox[cls_idx], bbox_arr])
                if segm_list is not None:
                    segm_list[cls_idx].append(masks[i])
        
        # Convert segm lists to arrays (old format: list of arrays per class)
        if segm_list is not None:
            segm = []
            for i in range(num_classes):
                if segm_list[i]:
                    # segm[i] should be an array of masks for this class
                    segm.append(np.array(segm_list[i]))
                else:
                    segm.append(None)
    else:
        # Fallback for old API (tuple format)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        if len(bbox[index]) == 0:
            continue
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist() # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            mask_val = None if segm is None or segm[index] is None else segm[index][max_obj]
            detected[classname].append((bbox[index][max_obj], mask_val))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': metadata,
        'details': {
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        }
    }


def main(args):
    full_results = []
    all_subfolders = sorted(os.listdir(args.imagedir))
    all_subfolders = all_subfolders[:5]
    for subfolder in tqdm(all_subfolders, desc="Evaluating images"):
        folderpath = os.path.join(args.imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        # Evaluate each image
        for imagename in os.listdir(os.path.join(folderpath, "samples")):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            result = evaluate_image(imagepath, metadata)
            full_results.append(result)
    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        full_results_hashable = [cast_dict_to_str(result) for result in full_results]
        pd.DataFrame(full_results_hashable).to_json(fp, orient="records", lines=True)
    with open(Path(args.outfile).parent / "results_readable.jsonl", "w") as fp:
        for result in full_results:
            json.dump(result, fp)
            fp.write("\n")

def cast_dict_to_str(d):
    return {k: json.dumps(v) if k in ["metadata", "details"] else v for k, v in d.items()}


if __name__ == "__main__":
    args = parse_args()
    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    THRESHOLD = float(args.options.get('threshold', 0.3))
    COUNTING_THRESHOLD = float(args.options.get('counting_threshold', 0.9))
    MAX_OBJECTS = int(args.options.get('max_objects', 16))
    NMS_THRESHOLD = float(args.options.get('max_overlap', 1.0))
    POSITION_THRESHOLD = float(args.options.get('position_threshold', 0.1))

    main(args)
