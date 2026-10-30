"""oxr_layout_infer.py — 调用 OXR layout 分析服务，产出中间 JSON。

OXR 使用 D-FINE Stage-2 B2 做版面分析，layout 服务是一个独立的 FastAPI。
本脚本只负责：把一批**原图**（png/jpg/jpeg）发给 layout 服务，取回每页的
boxes/labels/scores，按页落盘为「中间 JSON」，供后续 oxr2labelme.py 转成
LabelMe 格式人工校验。

坐标说明（重要）：
- 服务端后处理已做 letterbox 还原，返回的 boxes 是**原图像素坐标系** [x1,y1,x2,y2]。
- 本脚本绝不预缩放、不做 960 映射、不做 ratio/pad 还原，直接搬运原图坐标。

HTTP 客户端：优先使用标准库 ``urllib.request`` 发送 multipart 请求，避免额外依赖。
（参考 oxr/model/layout_client.py 用 httpx 异步实现；这里为独立可用改为同步、
零三方依赖版本。若未来需要复用官方客户端，可改为 import 该模块。）

端点约定（来自 services/layout_analyze/dfine_la/confidence.py::format_predictions）：
- POST {url}/inference
- multipart 上传多张 PNG，字段名 ``images``，每个文件
  ``(f"page-{i}.png", png_bytes, "application/octet-stream")``
- 响应 JSON::

      {"status":"success","params":{},"data":{
          "labels":[[int,...],...],
          "boxes":[[[x1,y1,x2,y2],...],...],
          "scores":[[float,...],...]}}

  外层 list 按页；每页内含该页的 labels/boxes/scores，三者等长且一一对应。

类别映射（与 oxr_bbox.py / layout_postprocess.py::LA_CLASS_ID_TO_BLOCK_TYPE 一致）：
0 Caption, 1 Footnote, 2 Formula, 3 Text, 4 Page-footer, 5 Page-header,
6 Picture, 7 Title, 8 Table, 9 Code Block, 10 Stamp, 11 Chemical Structure
（注意：旧模型只有 12 类，不会有 12=Author；但本脚本要把任意整数 label 透传为
对应 BlockType 名，未知 id 用 ``f"class_{id}"`` 兜底，不崩溃。）
⚠ 未来若 OXR 增加 Author 等类别（如 12=Author），此处映射常量需同步更新。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# 类别映射常量（内置于脚本，避免 import OXR 内部导致的依赖/循环问题）
# ⚠ 若 OXR 增加新类别（例如 12=Author），此处必须同步更新！
# ---------------------------------------------------------------------------
LA_CLASS_ID_TO_NAME: dict[int, str] = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "Text",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Title",
    8: "Table",
    9: "Code Block",
    10: "Stamp",
    11: "Chemical Structure",
}


def class_id_to_name(label_id: int) -> str:
    """整数类 id -> 类名。未知 id 用 ``class_{id}`` 兜底，不抛异常。"""
    return LA_CLASS_ID_TO_NAME.get(label_id, f"class_{label_id}")


# ---------------------------------------------------------------------------
# 图像编码（优先 cv2，缺失则退化为 Pillow，再缺失则报错）
# ---------------------------------------------------------------------------
def encode_image_to_png(image_path: str) -> bytes:
    """读取一张图片并返回 PNG 编码字节（BGR 原图直接编码，与 layout_client 一致）。"""
    try:
        import cv2  # type: ignore

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2.imread 返回 None，无法读取图像: {image_path}")
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError(f"cv2.imencode 失败: {image_path}")
        return buf.tobytes()
    except ImportError:
        # 无 cv2 时退化为 Pillow
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                import io

                b = io.BytesIO()
                rgb.save(b, format="PNG")
                return b.getvalue()
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise RuntimeError(
                "需要 cv2 或 Pillow 之一来编码图像，但两者都未安装。"
            ) from exc


def get_image_size(image_path: str) -> tuple[int, int]:
    """返回 (width, height) 原图像素尺寸。"""
    try:
        import cv2  # type: ignore

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2.imread 返回 None: {image_path}")
        h, w = img.shape[:2]
        return int(w), int(h)
    except ImportError:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            return int(im.width), int(im.height)


# ---------------------------------------------------------------------------
# 同步 multipart 上传（标准库 urllib，零三方依赖）
# ---------------------------------------------------------------------------
def _multipart_boundary() -> str:
    return f"----oxrboundary{uuid.uuid4().hex}"


def _multipart_body(files: list[tuple[str, bytes]], boundary: str) -> bytes:
    """构造 multipart/form-data 请求体。files: [(field_name, png_bytes), ...]"""
    crlf = b"\r\n"
    parts: list[bytes] = []
    for idx, (_, png_bytes) in enumerate(files):
        # 字段名固定为 images，文件名 page-{i}.png，content-type 为 octet-stream
        disp = f'form-data; name="images"; filename="page-{idx}.png"'
        head = (
            b"--" + boundary.encode("utf-8") + crlf
            + b"Content-Disposition: " + disp.encode("utf-8") + crlf
            + b"Content-Type: application/octet-stream" + crlf + crlf
        )
        parts.append(head)
        parts.append(png_bytes)
        parts.append(crlf)
    parts.append(b"--" + boundary.encode("utf-8") + b"--" + crlf)
    return b"".join(parts)


def request_layout_inference(
    url: str, image_paths: list[str]
) -> list[dict[str, Any]]:
    """对一批图片做同步 layout 推理，返回解析后的「每页结果」list。

    返回值结构（已对齐响应 JSON 的 data 字段，按页）：
        [ {"labels":[...], "boxes":[[x1,y1,x2,y2],...], "scores":[...]}, ... ]

    异常会向上抛出，由调用方决定跳过该批或整体报错（绝不静默产出空结果）。
    """
    import urllib.request

    if not image_paths:
        return []

    encoded: list[tuple[str, bytes]] = []
    for p in image_paths:
        encoded.append(("images", encode_image_to_png(p)))

    boundary = _multipart_boundary()
    body = _multipart_body(encoded, boundary)
    endpoint = url if url.rstrip("/").endswith("/inference") else f"{url.rstrip('/')}/inference"
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header(
        "Content-Type", f"multipart/form-data; boundary={boundary}"
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 - 网络/服务错误需上抛
        raise RuntimeError(
            f"layout 服务请求失败 url={endpoint} images={len(image_paths)}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"layout 服务返回非 JSON 内容: {exc}\nraw={raw[:200]!r}") from exc

    if payload.get("status") != "success":
        raise RuntimeError(f"layout 服务返回非 success 状态: {payload}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"layout 响应缺少 data 字段: {payload}")

    labels = data.get("labels")
    boxes = data.get("boxes")
    scores = data.get("scores")
    if not (isinstance(labels, list) and isinstance(boxes, list) and isinstance(scores, list)):
        raise RuntimeError(f"layout 响应 data 结构异常: {data}")

    # 每页一个 dict；若某页缺失则补空，避免索引错位
    n_pages = max(len(labels), len(boxes), len(scores))
    pages: list[dict[str, Any]] = []
    for i in range(n_pages):
        pages.append(
            {
                "labels": labels[i] if i < len(labels) else [],
                "boxes": boxes[i] if i < len(boxes) else [],
                "scores": scores[i] if i < len(scores) else [],
            }
        )
    return pages


# ---------------------------------------------------------------------------
# 中间 JSON 产出
# ---------------------------------------------------------------------------
def build_page_intermediate(
    source_name: str,
    width: int,
    height: int,
    page: dict[str, Any],
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """把单页布局结果（labels/boxes/scores）整理成中间 JSON dict。"""
    out_boxes: list[dict[str, Any]] = []
    labels = page.get("labels", [])
    boxes = page.get("boxes", [])
    scores = page.get("scores", [])
    for idx in range(min(len(labels), len(boxes), len(scores))):
        label_id = int(labels[idx])
        box = boxes[idx]
        score = float(scores[idx])
        if score_threshold > 0.0 and score < score_threshold:
            continue
        # box 已是原图像素坐标 [x1,y1,x2,y2]，直接搬运，不缩放
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        out_boxes.append(
            {
                "label_id": label_id,
                "label": class_id_to_name(label_id),
                "bbox": [x1, y1, x2, y2],
                "score": score,
            }
        )
    return {
        "source": source_name,
        "width": int(width),
        "height": int(height),
        "boxes": out_boxes,
    }


def collect_images(input_path: str) -> list[str]:
    """收集输入图片路径。支持单文件或目录（按文件名排序）。"""
    exts = (".png", ".jpg", ".jpeg")
    if os.path.isfile(input_path):
        if input_path.lower().endswith(exts):
            return [input_path]
        raise ValueError(f"输入文件不是支持的图片类型: {input_path}")
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(exts)
        ]
        return sorted(files)
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def run(
    input_path: str,
    url: str,
    out_dir: str,
    score_threshold: float = 0.0,
    batch_size: int = 8,
) -> int:
    """执行整批推理并落盘中间 JSON。返回成功处理的页数。"""
    images = collect_images(input_path)
    if not images:
        print(f"[warn] 未找到任何图片: {input_path}", file=sys.stderr)
        return 0

    os.makedirs(out_dir, exist_ok=True)
    total = 0
    # 分批请求；批内按页对齐返回顺序
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        try:
            pages = request_layout_inference(url, batch)
        except Exception as exc:  # noqa: BLE001
            # 失败处理：打印错误并跳过该批次（不静默产出空结果）
            print(
                f"[error] 批次推理失败（跳过 {len(batch)} 张，起始={start}）: {exc}",
                file=sys.stderr,
            )
            continue

        if len(pages) != len(batch):
            print(
                f"[warn] 返回页数({len(pages)})与请求张数({len(batch)})不一致，按最小对齐处理",
                file=sys.stderr,
            )

        for j, img_path in enumerate(batch):
            if j >= len(pages):
                print(f"[warn] 缺少第 {j} 页结果: {img_path}", file=sys.stderr)
                continue
            src_name = os.path.basename(img_path)
            w, h = get_image_size(img_path)
            inter = build_page_intermediate(
                src_name, w, h, pages[j], score_threshold=score_threshold
            )
            out_name = os.path.splitext(src_name)[0] + ".json"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(inter, f, ensure_ascii=False, indent=2)
            total += 1
            print(f"[ok] 已写出 {out_path}（{len(inter['boxes'])} 个框）")

    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="调用 OXR layout 分析服务，把原图转成中间 JSON（每页一个）。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="图片目录（或单张图片），支持 png/jpg/jpeg，按文件名排序。",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8081",
        help="layout 服务 base url（默认 http://127.0.0.1:8081）。",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出目录（每页一个 json，文件名与原图对应，如 page_001.png -> page_001.json）。",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="过滤低于该分的候选（默认 0.0 表示全留）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每次请求打包几张图（默认 8）。",
    )
    args = parser.parse_args(argv)

    n = run(
        input_path=args.input,
        url=args.url,
        out_dir=args.out,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
    )
    print(f"[done] 共成功处理 {n} 页 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
