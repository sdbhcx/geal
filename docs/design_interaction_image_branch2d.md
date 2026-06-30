# 设计:交互图注入 Branch2D(训练期知识注入)

> 目标:用 PIAD 自带的**人-物交互图**为 Stage 1 的 2D teacher 注入 affordance 定位/几何知识,从而改善"稀疏点云 → 高斯渲染几何退化 → DINO 特征不可靠"的问题。
> 关联代码:`model/branch_2d.py`(模型)、`dataset/piad.py` + `dataset/piad_process.py`(数据)、`scripts/train_stage1.py`(训练)。

---

## 0. 三个定性结论(决定了设计形态)

1. **只在 Stage 1、只在训练期使用**。推理时不存在"配套交互图",所以这是**训练期知识注入**,不是测试输入。Stage 2(3D 蒸馏)完全不动,自动受益于更好的 2D teacher。
2. **不做像素对齐**。交互图与点云:不同实例、含人/背景、无相机位姿、行数都对不上(Img 4149 vs Point 5999)。唯一可靠关联是路径里的 `(class, affordance)`。因此改做**文本条件下的"affordance 区域级"特征对齐**。
3. **Phase 1 无需额外标注**即可跑;有人-物接触伪标签时再加监督(Phase 2)。

---

## 1. 数据对应关系(已确认)

- `Img_{split}.txt` 与 `Point_{split}.txt` **非行对齐**,是两份独立清单。
- 图路径自带类别与 affordance:`{root}/Seen/Img/Train/{Class}/{affordance}/Img_*.jpg`。
- 已实现(数据层,零破坏):
  - `piad_process.build_image_index` → 生成旁路 `{setting}_{split}_img_index.pkl`,内容为 `{(class, affordance): [img_path, ...]}`,主 pkl 不动。
  - `PiadDataset(use_image=True)` → 按样本 `(class, affordance)` 从图池采一张交互图(train 随机 / test 取首张,缺失回退同类),返回 7-tuple 多出 `image`(已 resize+normalize)。

---

## 2. 当前 Stage 1 数据流(精确维度;config: res=112, emb=512, V=12)

```
点云 xyz [B,3,N]
  └─ 高斯渲染 ─► 渲染图 [B·12, 3,112,112]
       └─ frozen DINOv2(patch14) ─► patch tokens [B·12, 64, 768]   (8×8=64)
            └─ dino_embed ─► fused_feat [B·12, 64, 512]
文本(12 视角问题) ─► RoBERTa+proj ─► text_embeds [B·12, 40, 512], mask [B·12,40]
  GAFM(text, fused_feat) ─► cross_modal [B·12, 64, 512]
  decoder ─► text_feat [B·12,40,512]
  attn = einsum(text_feat, fused_featᵀ) ─► [B·12, 64]     ← 每个渲染视角的 affordance 权重
  reshape + upsample + sigmoid ─► attn_map [B·12, 1,112,112]
  损失: BCE(attn_map, 渲染GT灰度图)
```

---

## 3. 新增:交互图支路(复用 frozen DINO + 同一文本 + 同一 GAFM)

```
交互图 image [B, 3,224,224]                       ← dataset use_image=True
  └─ frozen DINOv2(patch14) ─► img tokens [B, 256, 768]   (16×16=256)
       └─ dino_embed(共享) ─► img_feat [B, 256, 512]
text_img [B,40,512]  ← text_embeds 重排 [B,12,40,512] 在视角维平均
  GAFM(text_img, img_feat)(共享) ─► [B, 256, 512]
  decoder(共享) ─► text_feat_img [B,40,512]
  attn_img = einsum ─► [B, 256]            ← 真实图上的 affordance 权重(副产物=图上热力图)
  z_img = softmax(attn_img) · img_feat ─► [B, 512]   ← 真实图 affordance 区域表征(teacher)
```

从渲染支路同样池化出逐样本表征:

```
z_render_view = softmax(attn[B·12,64]) · fused_feat[B·12,64,512] ─► [B·12, 512]
z_render      = mean over 12 views ─► [B, 512]     ← 跨视角聚合的 affordance 区域表征(student)
```

**关键**:两支路共享所有可训练模块(dino_embed / GAFM / decoder),DINO 冻结;只在**池化后的区域级向量**上对齐,从根本上避开"无像素对应"。

---

## 4. Loss 设计

### Phase 1(无需额外标注)— 区域级对齐,注入真实图知识

批内 InfoNCE(防塌缩),teacher 侧 detach:

```
L_align = InfoNCE(z_render, sg(z_img))
        = -1/B Σ_i log  exp(cos(z_render_i, z_img_i)/τ)
                        ─────────────────────────────────
                         Σ_j exp(cos(z_render_i, z_img_j)/τ)
```

- `sg` = stop-grad:真实图几何更可信,作为固定 teacher,让渲染支路单向靠近(更稳)。
- InfoNCE 而非纯 cosine:保证 `z_render_i` 匹配自身 `z_img_i` 胜过他者,避免向量塌缩。
- τ ≈ 0.07。

**Stage 1 总损失**:

```
L = BCE(attn_map, render_GT)  +  λ_align · L_align        (λ_align ≈ 0.1 ~ 0.5)
```

### Phase 2(可选)— 给真实图支路加监督,强化 teacher

```
L += λ_img · BCE(upsample(attn_img), contact_mask)
```

接触伪标签来源:对交互图离线跑 HOI/接触检测或 SAM+人物框,再扩一个旁路文件。

---

## 5. 代码落点(Stage 1 only)

| 改动 | 文件 | 内容 |
|---|---|---|
| 传图 | `scripts/train_stage1.py` `build_dataloader` | `PiadDataset(..., use_image=True)`;训练循环解包加 `image` 并 `.to(device)` |
| 图支路 | `model/branch_2d.py` | 新增 `_encode_image_tokens(image)`(复用 `self.dino_model`/`dino_embed`)+ `_affordance_pool(attn, feat)`;`forward` 的 `stage1` 分支多算 `z_img`/`z_render`,多返回二者 |
| 损失 | `model/branch_2d.py` 或 `utils/loss.py` | `info_nce(z_render, z_img)`;train 里 `loss = bce + λ·align` |
| 开关 | `config/train_stage1.yaml` `model_2d` | `use_image: true`、`img_align_weight: 0.2`、`img_size: 224`、`temp: 0.07` |

> 推理 / Stage 2 / evaluate **全部不变**:`use_image=False` 时 forward 不走图支路,返回值与现状一致。

---

## 6. 设计取舍(诚实标注)

- **为什么区域级而非像素级**:交互图含人/背景、不同实例、无位姿;只有池化到 affordance 区域的表征才有可比性。这正对应"几何知识注入"——渲染来自稀疏点云、几何退化,靠真实图的有效表征把渲染支路的 text-conditioned 表征往正确方向拉。
- **为什么 detach teacher**:让 student(渲染)单向靠近更稳;若想双向一致可去掉 `sg`,但要监控塌缩。
- **z_img 的弱点**:`attn_img` 此时无监督,可能聚焦到"人"而非物体接触区。两个缓解:(a) 预处理时用 SAM 把物体抠出来再喂 DINO(推荐,便宜);(b) 上 Phase 2 接触监督。**这是整套方案最需要实验验证的一环。**

---

## 7. 落地顺序建议

1. 数据机重跑 `piad_process.py` 生成 `*_img_index.pkl`(已实现)。
2. 验证 `PiadDataset(use_image=True)` 能正常出图。
3. 决策点:`z_img` 是否先 SAM 抠物体。
4. 实现 Phase 1(图支路 + InfoNCE + config 开关),Stage 1 跑通,对比 IOU/SIM/MAE。
5. 视效果决定是否上 Phase 2(接触伪标签监督)。

---

## 8. SAM teacher 实现与修订(2026-06,已落地)

### 8.1 背景:第一版 SAM 尝试为何无效

第一版把 SAM 的 **256 维图像嵌入**(`predictor.get_image_embedding()`,`[256,H/16,W/16]`)直接当 teacher 输入,经 `sam_proj` 投影到 `llm_dim`。日志对比(last-10-epoch 均值,与 §6(a) 的预期相反)显示 SAM ≈ 无SAM、甚至略差,根因有二:

1. **`sam_proj` 从未被训练(bug)**。它在 `forward` 里惰性创建(`if not hasattr(self,'sam_proj')`),而 `build_optimizer()` 早已在第一次 forward **之前**锁定参数组 → `sam_proj` 进不了 optimizer,恒为随机初始化。叠加 teacher 侧 `z_img.detach()`,该投影**两头都拿不到梯度**,teacher 退化成「SAM 特征过一个随机固定线性层」的噪声目标。
2. **SAM 嵌入是错的特征类型**。`get_image_embedding()` 面向可提示分割(物体边界/可分割性),不是语义 affordance 线索;而渲染支路 teacher 需要的是 DINOv2 那种语义表征。

### 8.2 修订一:`sam_proj` 注册到 `__init__`

`model/branch_2d.py`:`sam_proj` 移入 `__init__`(维度取 `cfg.sam_feat_dim`,默认 256),在 `build_optimizer()` 前注册为子模块 → 真正进 optimizer。`_encode_image_tokens` 的路由判据改为 `image.shape[1] == self.sam_feat_dim`。

### 8.3 修订二:SAM mask 抠前景 → masked RGB → DINO(默认路径)

落实 §6(a)。不再喂 256 维嵌入,而是**用 SAM 把前景物体抠出来、背景置零,得到 masked RGB**,3 通道天然走已训练的 **DINOv2 + `dino_embed`** 路径——`dino_embed`/`dino_embed_norm` 本就被渲染支路 BCE 训练(§2 Step 2 共享),所以 teacher 用的是**已训练的语义编码器**,既绕开 8.1 的死结,又去掉了背景/人物噪声。

```
交互图 ─► SAM(bbox 作 box prompt)分割前景 ─► 背景置零 + 裁剪到 bbox + resize
       ─► masked RGB [3,224,224] (uint8, 离线缓存)
训练时加载 ─► /255 + ImageNet 归一化 ─► frozen DINOv2 ─► dino_embed(已训练) ─► img_feat[B,256,512]
```

- 用 **bbox 作 prompt**(数据集 `Bounding_Box/` 已有)比自动分割更可靠;无 bbox 时回退整图。
- 离线缓存,避免训练时实时跑 SAM。两种模式存到不同文件,互不覆盖。

### 8.4 代码落点

| 改动 | 文件 | 内容 |
|---|---|---|
| 修 bug | `model/branch_2d.py` | `sam_proj` 移入 `__init__`(`sam_feat_dim`);路由判据改 `== self.sam_feat_dim` |
| 抠前景 | `dataset/preprocess_sam.py` | 新增 `extract_masked_rgb(image, bbox)`;`sam.mode` 选 `masked_rgb`/`feature`;存 `{split}_sam_masked_rgb_dict.pt` |
| 加载 | `dataset/piad.py` | 新增 `sam_mode`;`masked_rgb` 下加载 uint8 RGB 并归一化为 `[3,H,W]` |
| 开关 | `config/train_stage1.yaml` + `scripts/train_stage1.py` | `dataset.sam_mode`、`dataset.sam_feature_dir`、`sam:` 预处理块 |

### 8.5 运行步骤

masked RGB 与旧 256-d 缓存格式不同,训练前需**重跑预处理**:

```bash
python3 dataset/preprocess_sam.py --config config/train_stage1.yaml   # 默认 mode=masked_rgb
```

生成 `<data_root>/sam_features/{train,test}_sam_masked_rgb_dict.pt` 后,确保 `dataset.category: piad`、`use_image: true`、`sam_mode: masked_rgb`,即可启动 Stage 1。

### 8.6 仍待验证

- masked RGB(本方案)vs 256-d 特征(已修 `sam_proj`)vs 无SAM RGB,三者 IOU/AUC/SIM/MAE 对比。
- 评估时建议报告 last-N epoch 均值±std 或多 seed:上一轮 SAM 与无SAM 的差异落在 run 间噪声(IOU std≈0.002–0.003)以内,单 epoch 快照会得出错误结论。
