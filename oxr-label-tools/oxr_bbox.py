"""OXR layout bounding-box helpers.

本模块只做"原图像素坐标 -> COCO [x, y, w, h]"的纯搬运，
**不做任何 960 缩放**（960 缩放完全交给训练端的 letterbox 后处理）。

重要约定（已与 OXR 服务端核对）：
- layout 服务 /inference 返回的 boxes 是已还原回原图的像素坐标 [x1, y1, x2, y2]。
- 训练端会自己用 letterbox 把原图缩放到 960。
- 因此标注/转换阶段绝不能预缩放坐标：所有中间格式（labelme json、COCO）
  都必须存原图像素坐标。本模块的转换函数只做
  [x1, y1, x2, y2] -> [x, y, w, h]（w = x2 - x1, h = y2 - y1）的搬运，
  不得做 ratio/pad 映射。

类别定义（id 顺序固定，不可更改）：
0 Caption, 1 Footnote, 2 Formula, 3 Text, 4 Page-footer, 5 Page-header,
6 Picture, 7 Title, 8 Table, 9 Code Block, 10 Stamp, 11 Chemical Structure,
12 Author（本次新增的作者块，含姓名+单位）。
"""

from __future__ import annotations

# 类别 id -> 名称（顺序固定，Author 固定为 id=12）
ID2LABEL: dict[int, str] = {
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
    12: "Author",
}

# 名称 -> id
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# 类别总数（12 原类 + Author = 13）
NUM_CLASSES: int = len(ID2LABEL)


def bbox_to_coco(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """把原图像素坐标 [x1, y1, x2, y2] 转为 COCO bbox [x, y, w, h]。

    仅做纯搬运（w = x2 - x1, h = y2 - y1），**不缩放**。
    坐标单位为原图像素，与 /inference 输出以及训练端输入保持一致。
    """
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
