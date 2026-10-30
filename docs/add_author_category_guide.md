# OXR 新增「Author」类别并实施 LoRA 微调 — 行动指南（修订版 v2）

> 基于 **2026-10-30 对 OXR 代码库的实际核对** 修订（v1 已纠正原指南偏差；v2 补充
> layout 服务输出结构核对结论 + 图片尺寸处理细则 + 三个新建工具的数据契约）。
> 本指南**不立即执行任何改动**，仅作为后续行动的权威依据。

---

> **版本管理（git，已就绪）**：本工作全部在独立分支管理，未污染 `main` / 官方 `origin`。
> - 分支：`feature/add-author-category`（已从 `main` 切出，跟踪 `mine` 远端）
> - 远端：`origin` = `https://github.com/SoMarkAI/OXR.git`（官方，只读）；
>   `mine` = `git@github.com:mir-of/OXR.git`（个人 fork，SSH，已推送）
> - 已提交：`6fa3e24 docs: add author-category action guide (v2, code-verified)`
> - 未纳入：原 `docs/add_author_category_lora_guide.md`（旧指南）；`docker/Dockerfile.allinone` 的无关改动保留不动
> - 提交约定：每完成一个阶段（代码接入 / 工具新建 / 训练闭环）单独 commit；只 `git add` 本次相关文件，避免误带无关改动
> - 推送：`git push -u mine feature/add-author-category`（已验证可通）

---


## 〇、代码现状核对（已核实）

| 项目 | 真实事实 |
|---|---|
| 模型 | D-FINE Stage-2 B2，输入 960×960，backbone HGNetv2-B2 |
| 服务 | `services/layout_analyze/dfine_la`（FastAPI，`POST /inference`） |
| 当前类别数 | **12 类**（id 0–11） |
| `oxr-label-tools/` 目录 | **不存在**（无 extract/to_coco/oxr_bbox） |
| `xycut_order._LABEL_MAP` | **12 项 `dict`**：`{BlockType.X: "下游标签"}` |
| `model.py` 默认 `num_classes` | `DFINE.__init__` 与 `load_model` 均默认 `12` |
| `decoder.py` 分类头 | 按 `num_classes` 动态建 `dec_score_head`/`enc_score_head`/`denoising_class_embed` |
| `metadata.json` | `num_classes=12`，`id2label` 0–11，位于 `/root/qj/models/oxr-model/layout_analysis/` |
| `config.py` 校验 | `load_metadata()` 对 `SUPPORTED_METADATA` 逐字段严格 `value != supported` 即报错 |
| 训练环境 | 远程系统 `python3` 无 torch；`dfine_la/requirements.txt` 仅推理依赖 |

### 现有 12 类别（id 0–11，固定顺序）
```
0 Caption   1 Footnote   2 Formula   3 Text      4 Page-footer
5 Page-header 6 Picture  7 Title     8 Table     9 Code Block
10 Stamp    11 Chemical Structure
```
新增 `Author` 作为第 **12** 类（id=12）。**类别顺序必须稳定，Author 固定末位。**

---

## 一、Layout 服务输出结构（已核对，v2 新增）

> 用途：阶段 1 用 OXR 推理出底稿前，必须先知道它吐出什么坐标空间，否则转换器会错。

### 调用链
```
LayoutAnalyzeClient.request_batch(encoded_png_bytes[])
   → POST {url}/inference  (multipart: images[])
   → 返回 JSON
parse_layout_batch(payload, image_sizes, score_threshold, nms_iou_threshold)
   → list[list[Block]]   （每页一组 Block）
```

### 服务端 `/inference` 响应契约
`services/layout_analyze/dfine_la/confidence.py::format_predictions` 返回：
```json
{
  "status": "success",
  "params": {},
  "data": {
    "labels": [ [int, ...], ... ],   // 每页一组类 id（0–11）
    "boxes":  [ [[x1,y1,x2,y2], ...], ... ],  // 每页一组 bbox
    "scores": [ [float, ...], ... ]  // 每页一组置信度
  }
}
```
- `boxes` 顺序与 `labels`/`scores` 一一对应。
- **坐标空间：原图像素 `[x1,y1,x2,y2]`（已还原，非 960）**。

### 为什么 boxes 是原图坐标（已追到后处理）
1. `preprocess.py::resize_and_pad` 把原图 letterbox 到 960（`ratio=960/max(w,h)`，`pad_left/pad_top`）。
2. `model/postprocessor.py` 的 `DFINEPostProcessor`（letterbox 模式）在输出时做逆变换：
   `bbox = (bbox_model - pad) / scale` → **还原回原图像素坐标**。
3. `runtime.py::health` 也自报 `preprocess_type: "resize_pad"`、`postprocess_mode: "model_letterbox"`。

**结论：OXR `/inference` 返回的 boxes 直接就是原图像素坐标，与输入图片尺寸一致。
`oxr2labelme.py` 转换时无需任何坐标逆变换，直接搬运即可。**

### Block 结构（解析后，`oxr/pipeline/models.py`）
```python
Block(idx:int, type:BlockType, bbox:[x1,y1,x2,y2], content:str, format:ContentFormat, layout_score:float|None)
```
- `type` 是 `BlockType` 枚举，由 `LA_CLASS_ID_TO_BLOCK_TYPE[label]` 映射（见 `layout_postprocess.py`）。
- 坐标 `bbox` 已是原图像素。

---

## 二、图片尺寸处理细则（D-FINE 训练，v2 新增，关键）

**核心结论：标注链路全程使用原图像素坐标，960×960 缩放完全交给 D-FINE 训练端 letterbox 处理。不要在标注/转换阶段预缩放。**

1. **原始素材**：PDF 先 rasterize 成图片（建议 ~200 DPI，长边 1500–2000px 为宜；过低分辨率伤小目标）。
2. **同一份图片两用**：既喂 OXR 推理出底稿，也喂 LabelMe 供人工改框——保证 bbox 坐标对齐。
3. **OXR 推理**：输入任意尺寸原图，输出原图像素 boxes（见上）。
4. **LabelMe 标注**：在原图上标，json 存原图坐标。作者区删 `Text` 框、画 `Author` 矩形框。
5. **labelme2coco**：COCO `bbox` 直接存**原图像素** `[x,y,w,h]`（从 labelme `[x1,y1,x2,y2]` 算），**不做 960 映射**。
6. **D-FINE 训练端**：配置 `image_size=960` + `resize_mode=letterbox`（非拉伸），框架自动
   等比缩放 + 灰边填充，并同步变换 bbox。若标注已预缩放会导致二次缩放错位。
7. **推理部署端**：`config.py` 的 `postprocess: "model_letterbox"` 已在模型内还原坐标，无需额外处理。

> 修正 v1 指南：v1 给的 `oxr_bbox.xywh_to_coco` 做了 960 letterbox 映射是错误的/多余的，
> 会让训练端二次缩放。正确实现见下方 `oxr_bbox.py` 约定（仅做原图→COCO 坐标搬运）。

---

## 三、目标与阶段拆解（同 v1）

目标：在 D-FINE Stage-2 B2 中新增 `Author`（作者块，含姓名+单位）类别，
通过 LoRA 微调在 12 类模型上完成训练与部署闭环。
标注语义：把原误标为 `Text` 的作者信息块，删除 `Text` 框并重标为 `Author` 矩形框。

阶段：0 代码接入 → 1 标注（OXR 底稿→LabelMe 改框）→ 2 转 COCO → 3 LoRA 训练 → 4 部署闭环。

---

## 四、行动步骤

### 阶段 0 · 对齐类别定义（改 4 处 + 新建工具，零风险）
1. **新建 `OXR/oxr-label-tools/oxr_bbox.py`**（官方缺失）：
   ```python
   # 仅做原图→COCO 坐标搬运，不做 960 缩放（缩放交给训练端 letterbox）
   ID2LABEL = {
       0:"Caption",1:"Footnote",2:"Formula",3:"Text",4:"Page-footer",
       5:"Page-header",6:"Picture",7:"Title",8:"Table",9:"Code Block",
       10:"Stamp",11:"Chemical Structure",12:"Author",
   }
   LABEL2ID = {v: k for k, v in ID2LABEL.items()}

   def bbox_to_coco(x1, y1, x2, y2):
       """原图像素 [x1,y1,x2,y2] -> COCO [x,y,w,h]（原图坐标，不缩放）。"""
       return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
   ```
2. **`dfine_la/config.py`**：`SUPPORTED_METADATA` 的 `num_classes:12→13`；`id2label` 末位加 `"12":"Author"`。
3. **`dfine_la/model/model.py`**：`DFINE.__init__` 与 `load_model` 默认 `num_classes=12→13`。
4. **`oxr/pipeline/models.py`**：`BlockType` 枚举加 `AUTHOR = "Author"`（置于 `CODE_BLOCK` 之后）。
5. **`oxr/pipeline/xycut_order.py`**：`_LABEL_MAP` 加 `BlockType.AUTHOR: "author"`（单独下游标签，参与阅读顺序）。
6. **前向验证（不训练）**：构造 `num_classes=13` 的 `DFINE`，用现有 `last.pth` 以 `strict=False` 加载
   （容忍分类头维度变化），跑通单张图前向，确认 13 维输出、无 RuntimeError。

### 阶段 1 · 数据标注（路 1：OXR 输出 → LabelMe，因官方工具缺失）

> 前提：你只有原始 PDF/图片（无现成标注），故先调 OXR 推理出全类底稿，再在 LabelMe 里改作者区。

1. **rasterize**：PDF → 图片（~200 DPI）。同一份图片既喂推理也喂标注。
2. **出底稿 `oxr_layout_infer.py`（已落地）**：
   - 调 OXR layout 服务 `POST {url}/inference`（multipart，字段名 `images`）。
   - **服务端点**：默认 `http://127.0.0.1:8081`（来自 `config.example.yaml` 的 `layout_analyze_model.url`，运行时可配；脚本 `--url` 可覆盖）。
   - 输入：原图文件；输出：每页 `labels/boxes/scores`（**原图像素**，服务端已 letterbox 还原）。
   - 落盘为中间 JSON（每页一个，含 `source/width/height/boxes[]`，box 为 `label_id/label/bbox[x1,y1,x2,y2]/score`）。
   - 用法：`python oxr-label-tools/oxr_layout_infer.py --input <图片目录> --url http://127.0.0.1:8081 --out <中间JSON目录> [--score-threshold 0.5 --batch-size 8]`
3. **转 LabelMe `oxr2labelme.py`（已落地）**：
   - 每个 Block → `shape`：`shape_type:"rectangle"`，`points:[[x1,y1],[x2,y2]]`，`label: BlockType 名`。
   - `imagePath` 指向原图；`imageHeight/Width` 用原图尺寸。
   - **坐标直接搬运，不做变换**（OXR 已还原原图）。
   - 输出：与原图同名的 `xxx.json`，放入 LabelMe 工程目录。
   - 用法：`python oxr-label-tools/oxr2labelme.py --input <中间JSON目录> --images <原图目录> --out <labelme_json目录>`
4. **人工改框（LabelMe GUI）**：
   - `Open Dir` 打开工程目录；找到作者区（标题下方：姓名+单位）原 `Text` 框 → Delete。
   - 按 `R` 画整个作者块矩形 → label 严格输入 `Author`（大小写/拼写须与 `ID2LABEL` 一致）。
   - 其余 `Text`/其他类框原样保留。漏检的作者区直接补画。

### 阶段 2 · 转 COCO（`oxr-label-tools/labelme2coco.py` 已落地）
- 读 labelme json 目录，按 `oxr_bbox.ID2LABEL`/`LABEL2ID` 与 `bbox_to_coco` 映射，
  产出标准 COCO：`images`/`annotations`/`categories`（共 13 类，`Author` id=12）。
- **bbox 存原图像素，不预缩放**（见第二节）。
  ```bash
  python oxr-label-tools/labelme2coco.py \
    --input <labelme_json_dir> --images <原图_dir> \
    --out oxr-label-tools/coco/instances_layout.json
  ```
- 建议按图清单切 train/val（同图不入双集）。

### 阶段 3 · 训练（上游 D-FINE + PEFT LoRA，需另搭环境）
1. 新建 venv/conda，装 torch、pycocotools、peft + 上游 D-FINE 训练框架。
2. 数据集 = `instances_layout.json`，`image_size=960`，`resize_mode=letterbox`，`num_classes=13`。
3. **LoRA 混合策略**：
   - LoRA 注入 backbone/encoder/decoder 的 `Linear`（自注意力、cross_attn、FFN）。
   - 分类头 `dec_score_head`/`enc_score_head`/`denoising_class_embed` **全量训练**（第 13 维无预训练）。
   - 冻结 backbone 底层（`freeze_at=3`），只训 encoder+decoder+分类头+LoRA。
4. 以现有 `last.pth` 基座初始化（新列随机初始化，旧列继承）。
5. 产出新 `last.pth`。

### 阶段 4 · 部署闭环
1. 更新 `metadata.json`：`num_classes:13`、`id2label` 加 `"12":"Author"`、`model_version` 递增（如 `1.1.0`）、重算 `checkpoint_size_bytes` 与 `checkpoint_sha256`。
2. 覆盖权重：`cp <训练产出>/last.pth /root/qj/models/oxr-model/layout_analysis/last.pth`（同步 metadata）。
3. 重启 OXR layout 服务/容器（`config.py` 严格校验须与 metadata 完全一致）。
4. 闭环验证：跑曾出错含作者文档，对比 `Author` 修正率；仍错回到阶段 1 迭代。

---

## 五、执行顺序

```
阶段0(代码接入 + 13类前向验证)
   ↓
阶段1(rasterize → OXR推理底稿 → oxr2labelme → LabelMe改Author框)
   ↓                                   ↓
阶段2(labelme2coco 转 COCO，原图坐标)
   ↓
阶段3(上游 D-FINE + LoRA 训练，另搭环境)
   ↓
阶段4(部署 + 闭环验证 → 迭代)
```

---

## 六、风险与注意
- **分类头无法纯 LoRA**：第 13 维必须全量训练。
- **metadata 严格校验**：`num_classes` 等须与 `metadata.json` 完全一致，否则拒绝加载。
- **坐标空间**：标注全程原图像素；960 letterbox 由训练端处理，禁止在标注/转换阶段预缩放。
- **训练环境**：远程 python 无 torch，需另建 venv/conda。
- **类别顺序稳定**：`Author` 固定 id=12；未来增类须 4 处定义与 metadata 同步位移。
- **标注工具链（自建）**：`oxr-label-tools/` 原不存在，现已新建四个工具（oxr_bbox/oxr_layout_infer/oxr2labelme/labelme2coco，commit 8050cd0），阶段 1+2 可直接使用。
- **xycut 映射**：`Author→"author"`，独立于 `_NO_MASK_LABELS`，参与阅读顺序。
- **OXR 输出坐标已为原图**：`oxr2labelme.py` 直接搬运，无需逆变换。

---

## 七、相关文件速查

| 用途 | 路径 | 现状 |
|---|---|---|
| 类别定义 | `OXR/oxr-label-tools/oxr_bbox.py` | ✅ 已落地 (8050cd0) |
| OXR 推理底稿 | `OXR/oxr-label-tools/oxr_layout_infer.py` | ✅ 已落地 (8050cd0) |
| OXR→LabelMe | `OXR/oxr-label-tools/oxr2labelme.py` | ✅ 已落地 (8050cd0) |
| LabelMe→COCO | `OXR/oxr-label-tools/labelme2coco.py` | ✅ 已落地 (8050cd0) |
| 模型定义 | `OXR/services/layout_analyze/dfine_la/model/model.py` | 默认 num_classes=12，待改 |
| 配置/校验 | `OXR/services/layout_analyze/dfine_la/config.py` | SUPPORTED_METADATA 待改 |
| 分类头 | `OXR/services/layout_analyze/dfine_la/model/decoder.py` | 动态随 num_classes，无需改 |
| 推理响应契约 | `OXR/services/layout_analyze/dfine_la/confidence.py::format_predictions` | 返回原图坐标 boxes |
| 后处理还原 | `OXR/services/layout_analyze/dfine_la/model/postprocessor.py` | letterbox 还原原图 |
| 预处理缩放 | `OXR/services/layout_analyze/dfine_la/preprocess.py::resize_and_pad` | 960 letterbox |
| 客户端 | `OXR/oxr/model/layout_client.py` | 现存，可复用 |
| 响应解析 | `OXR/oxr/pipeline/postprocess/layout_postprocess.py` | 现存，可参考 |
| 类别枚举 | `OXR/oxr/pipeline/models.py` | BlockType 待加 AUTHOR |
| 标签映射 | `OXR/oxr/pipeline/xycut_order.py` | _LABEL_MAP 待加 AUTHOR:"author" |
| 权重/元数据 | `/root/qj/models/oxr-model/layout_analysis/` | last.pth + metadata.json（12 类） |
