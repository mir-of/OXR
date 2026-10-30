"""oxr2labelme.py — 把 oxr_layout_infer.py 产出的中间 JSON 转成 LabelMe 格式。

用途：在 LabelMe GUI 中打开原图 + 本脚本产出的 json，人工把「作者区 Text 框」
改标为 Author 框（OXR 当前模型只分 12 类、无 Author，需人工补标）。

输入：每页一个中间 JSON（来自 oxr_layout_infer.py 的 --out 目录），结构示例::
    {
      "source": "<原图文件名>",
      "width": <原图像素宽>,
      "height": <原图像素高>,
      "boxes": [ {"label_id": int, "label": "<类名>", "bbox":[x1,y1,x2,y2], "score":float}, ... ]
    }

输出：每页一个 LabelMe json（与原图同名），结构::
    {
      "version": "5.3.1",
      "flags": {},
      "imagePath": "<原图文件名>",
      "imageData": null,
      "imageHeight": <h>,
      "imageWidth": <w>,
      "shapes": [ {"label": "<类名>", "points":[[x1,y1],[x2,y2]],
                   "group_id": null, "shape_type": "rectangle", "flags": {}}, ... ]
    }

坐标说明（重要）：
- 中间 JSON 的 bbox 已经是**原图像素坐标** [x1,y1,x2,y2]（oxr_layout_infer 已直接搬运）。
- 本脚本仅搬运坐标，绝不缩放、不做 960 映射。

类别兜底：label 直接取自中间 JSON 的 "label" 字段（已是类名，如 "Text"）；
若中间 JSON 因未知 id 使用了 ``class_{id}`` 兜底名，这里保持一致透传。

⚠ 与 oxr_layout_infer.py 共享同一套类别语义；若 OXR 增加 Author 等类别，
oxr_layout_infer.py 的映射常量更新后，本脚本无需改动（类名已透传）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


LABELME_VERSION = "5.3.1"


def convert_one(inter: dict[str, Any], image_path_name: str) -> dict[str, Any]:
    """把单个中间 JSON dict 转成 LabelMe json dict。"""
    width = int(inter.get("width", 0))
    height = int(inter.get("height", 0))

    shapes: list[dict[str, Any]] = []
    for box in inter.get("boxes", []):
        bbox = box.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        # label 直接取自中间 JSON（已是类名，如 "Text"）；未知 id 用 class_{id} 兜底
        label = box.get("label", "Unknown")
        shapes.append(
            {
                "label": str(label),
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "imagePath": image_path_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "shapes": shapes,
    }


def run(input_dir: str, images_dir: str, out_dir: str) -> int:
    """转换整目录中间 JSON -> LabelMe json。返回成功转换页数。"""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"中间 JSON 目录不存在: {input_dir}")

    os.makedirs(out_dir, exist_ok=True)
    json_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(".json")
    )
    if not json_files:
        print(f"[warn] 未在 {input_dir} 找到任何 json", file=sys.stderr)
        return 0

    total = 0
    for jf in json_files:
        in_path = os.path.join(input_dir, jf)
        with open(in_path, "r", encoding="utf-8") as f:
            inter = json.load(f)

        # imagePath：指向原图文件名（优先用中间 JSON 的 source，否则用 json 同名）
        source = inter.get("source")
        if source:
            img_name = os.path.basename(source)
        else:
            img_name = os.path.splitext(jf)[0]
            # 尝试在 images_dir 中找同名的原图（仅用于 imagePath 命名，不强制存在）
            for ext in (".png", ".jpg", ".jpeg"):
                cand = os.path.join(images_dir, img_name + ext)
                if os.path.isfile(cand):
                    img_name = img_name + ext
                    break

        labelme = convert_one(inter, img_name)
        out_name = os.path.splitext(jf)[0] + ".json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelme, f, ensure_ascii=False, indent=2)
        total += 1
        print(f"[ok] 已写出 {out_path}（{len(labelme['shapes'])} 个 shape）")

    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 oxr_layout_infer.py 产出的中间 JSON 转成 LabelMe 格式 json（每页一个）。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="中间 JSON 目录（oxr_layout_infer.py 的 --out）。",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="原图目录，用于填 LabelMe 的 imagePath（LabelMe 打开时需要同目录有原图）。",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出 LabelMe json 目录。",
    )
    args = parser.parse_args(argv)

    n = run(input_dir=args.input, images_dir=args.images, out_dir=args.out)
    print(f"[done] 共转换 {n} 页 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
