#!/usr/bin/env python3
"""LabelMe 标注目录 -> 标准 COCO json 转换脚本。

坐标约定（与 oxr_bbox 一致）：
- 本脚本**只使用原图像素坐标**，不做 960 缩放（960 缩放由训练端 letterbox 处理）。
- LabelMe json 中的 points 必须是原图像素坐标 [x1, y1], [x2, y2]。
- COCO bbox 由 oxr_bbox.bbox_to_coco 生成：原图 [x1, y1, x2, y2]
  -> [x, y, w, h]（w = x2 - x1, h = y2 - y1）。

用法示例：
    python3 labelme2coco.py \
        --input /path/to/labelme_jsons \
        --images /path/to/origin_images \
        --out /path/to/output/coco.json

若 image 目录下找不到某张图，则回退使用 json 内的 imageHeight/imageWidth。
若某个 shape 的 label 不在 oxr_bbox.LABEL2ID 中，打印警告并跳过该 shape（不崩溃）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 保证同目录下的 oxr_bbox 可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oxr_bbox import ID2LABEL, LABEL2ID, bbox_to_coco  # noqa: E402


def _read_image_size(images_dir: str, image_path_field: str):
    """尝试从图片文件读取 (height, width)；失败返回 (None, None)。"""
    if not image_path_field:
        return None, None
    base = os.path.basename(image_path_field)
    cand = os.path.join(images_dir, base)
    if not os.path.isfile(cand):
        return None, None
    try:
        from PIL import Image  # 可选依赖，尽量不强制

        with Image.open(cand) as im:
            w, h = im.size
            return h, w
    except Exception:
        return None, None


def convert(input_dir: str, images_dir: str, out_path: str) -> dict:
    """把 input_dir 下的所有 labelme json 转成一个 COCO dict 并写入 out_path。"""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input 目录不存在: {input_dir}")

    images = []
    annotations = []
    categories = [{"id": cid, "name": name} for cid, name in ID2LABEL.items()]

    image_id = 0
    ann_id = 0

    json_files = sorted(
        fn for fn in os.listdir(input_dir) if fn.lower().endswith(".json")
    )

    for fn in json_files:
        json_path = os.path.join(input_dir, fn)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 跳过无法解析的 json: {json_path} ({e})")
            continue

        # 图片尺寸优先从 json 自带字段拿，回退从图片文件读
        h = data.get("imageHeight")
        w = data.get("imageWidth")
        if not h or not w:
            h2, w2 = _read_image_size(images_dir, data.get("imagePath", ""))
            h = h or h2
            w = w or w2
        if not h or not w:
            print(f"[WARN] 无法确定图片尺寸，跳过: {json_path}")
            continue

        file_name = data.get("imagePath") or fn
        if isinstance(file_name, (list, tuple)):
            file_name = file_name[0] if file_name else fn
        file_name = os.path.basename(str(file_name))

        image_id += 1
        images.append(
            {"id": image_id, "file_name": file_name, "height": int(h), "width": int(w)}
        )

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in LABEL2ID:
                print(f"[WARN] 未知 label 跳过: {label!r} (文件 {fn})")
                continue

            points = shape.get("points")
            if not points or len(points) < 2:
                print(f"[WARN] shape 缺少 points，跳过 (文件 {fn}, label {label!r})")
                continue

            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            bbox = bbox_to_coco(x1, y1, x2, y2)
            w_b, h_b = bbox[2], bbox[3]
            area = float(w_b * h_b)

            seg = [x1, y1, x2, y1, x2, y2, x1, y2]

            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(LABEL2ID[label]),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [seg],
                }
            )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(
        f"[INFO] 完成: {len(images)} 张图, {len(annotations)} 个标注 -> {out_path}"
    )
    return coco


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LabelMe 标注目录 -> COCO json（原图像素坐标，不做 960 缩放）"
    )
    parser.add_argument("--input", required=True, help="labelme json 目录")
    parser.add_argument(
        "--images", required=True, help="原图目录（用于读尺寸/文件名；json 自带尺寸时作回退）"
    )
    parser.add_argument("--out", required=True, help="输出 COCO json 路径")
    args = parser.parse_args(argv)

    convert(args.input, args.images, args.out)


if __name__ == "__main__":
    main()
