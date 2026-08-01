# GEAL V1 改动方案：连续局部 3D Tokenizer

> 基于：`D:\study\deep-learning\paper\2-Ai work\GEAL - Generalizable 3D Affordance Learning with Cross-Modal Consistency\GEAL 改进方案.md` 的 V1 版本。
>
> 本文是面向当前代码仓库的**可执行改动方案**，不是论文事实或已验证的实验结论。V1 只验证“显式局部连续 3D token 是否比现有 PointNet++ 层级特征更有利于未见类别 affordance grounding”，不同时引入 V2 的语言条件 token 交互、V3 动态 token 数、V4 token-level 2D–3D consistency、V5 VQ 或 V6 鲁棒一致性训练。

---

## 1. V1 目标与边界

### 1.1 目标

在 GEAL 当前 3D 分支中并联一个几何条件的局部 tokenizer：

```text
原始点云
  → FPS 选局部中心
  → KNN 构造固定邻域
  → 相对位置编码
  → 局部 MLP + pooling
  → 连续局部 3D tokens
  → token-to-point 插值传播
  → 与 PointNet++ dense feature 门控融合
  → 原有文本融合与逐点 affordance decoder
```

核心约束：

1. PointNet++ 主干保留，第一版不替换。
2. 2D Branch、DINOv2、RoBERTa、GAFM、Gaussian renderer 和原有主损失不改。
3. tokenizer 不读取语言，避免把 V1 与 V2 混在一起。
4. 每个 token 保留 `center_point_id`、`neighbor_point_ids` 和局部坐标信息，保证能够回到逐点预测。
5. 推理时只使用点云和文本，不能新增交互图或 2D 输入。
6. 通过门控残差融合控制新分支的影响，允许 `enabled: false` 完全复现旧模型结构。

### 1.2 不纳入 V1 的内容

- 不加入 token-level 2D–3D 对比损失；这属于 V4。
- 不把 token 数做成完全动态；先做固定 `M` 的连续 token。
- 不引入 VQ/codebook、LLM、GRPO 或 preference learning。
- 不删除 PointNet++，不把 token 直接作为最终输出。
- 不把现有 IAM/ADM、交互图 teacher、per-Gaussian render consistency 同时叠加到 V1 主实验中。

---

## 2. 当前代码基线与关键实现事实

### 2.1 3D 主干现状

主要文件：`model/branch_3d.py`

- `Branch3D.forward()` 接收：
  - `text`：当前 batch 的文本问题列表；
  - `xyz`：`[B, 3, N]`，数据集通常为 2048 点。
- PointNet++ 层级输出来自 `PointEncoder`：
  - `feat_lvl0`：原始点，约 `N=2048`；
  - `feat_lvl1`：`N=512`；
  - `feat_lvl2`：`N=128`；
  - `feat_lvl3`：`N=64`。
- 现有 FP 路径最终得到：
  - `fused_feat`：`[B, 512, 2048]`；
  - `affordance_map`：`[B, 2048]`。
- 训练模式额外返回：

```text
(pred_3d, downsampled_feat, gaussian_aff, patch_feat_3d, fused_feat)
```

关键位置：

- `model/branch_3d.py:123`：`forward` 入口；
- `model/branch_3d.py:143-145`：PointNet++ 编码；
- `model/branch_3d.py:151-161`：FP 解码到 2048 点；
- `model/branch_3d.py:163-170`：可选多尺度融合；
- `model/branch_3d.py:172-183`：Transformer 文本解码和逐点 affordance 预测；
- `model/branch_3d.py:185-198`：训练期特征和辅助头输出。

### 2.2 可复用的几何算子

主要文件：`model/pointnet2_utils.py`

已经存在：

- `square_distance(src, dst)`：批量平方距离；
- `index_points(points, idx)`：批量索引；
- `farthest_point_sample(xyz, npoint)`：FPS；
- `query_ball_point(radius, nsample, xyz, new_xyz)`：半径邻域；
- `PointNetFeaturePropagation`：3-NN 逆距离插值；
- `PointNetSetAbstractionMsg`：PointNet++ 多尺度分组。

V1 应复用上述实现，新增的 KNN 只需要在 `square_distance` 基础上用 `topk` 实现，以保证每个中心始终获得固定数量邻居，不受半径内点数不足影响。

### 2.3 训练与评估现状

主要文件：`scripts/train_stage2.py`、`scripts/evaluation.py`。

- Stage 2 训练入口：`scripts/train_stage2.py:409-582`。
- 模型创建：`scripts/train_stage2.py:428-429`。
- 原有点级主监督：`HM_Loss(pred_3d, label)`。
- 验证指标：IoU、AUC、SIM、MAE；评估脚本还按类别和 affordance 聚合。
- 当前 `config/train_stage2.yaml` 的 `use_iam_adm: true`。V1 不应与 IAM/ADM 混做首轮因果验证，应另建 V1 配置并将其关闭。

注意：当前 `train_stage2.py` 在训练循环中会无条件调用 `model_2d(question, point, feat_3d, gaussian_aff)`。V1 不需要修改 2D teacher，但建议在 V1 实验配置中保留原有 2D→3D MSE 路径，并单独记录其开销；不要把“是否调用 2D 分支”的优化与 tokenizer 改动混在同一组实验中。

---

## 3. 推荐的 V1 架构

### 3.1 模块位置

在 `Branch3D` 中使用原始几何点坐标建立 tokenizer，并在已有 `fused_feat` 形成之后进行 token-to-point 传播和融合：

```text
xyz [B,3,N]
  ├─ PointEncoder → FP → fused_feat [B,512,N]
  └─ Local3DTokenizer → token_feat [B,512,M]
                              ↓
                       TokenToPoint [B,512,N]
                              ↓
        fused_feat + γ · token_point_feat
                              ↓
                  GAFM 后续 decoder / affordance map
```

推荐放置点：

```python
# branch_3d.py，现有 Step 5 多尺度融合之后、Step 6 Transformer 解码之前
if self.use_local_tokenizer:
    token_feat, token_meta = self.local_tokenizer(xyz.transpose(1, 2), fused_feat)
    token_point_feat = self.token_to_point(xyz.transpose(1, 2), token_feat, token_meta)
    fused_feat = self.token_fusion(fused_feat, token_point_feat)
```

这样做的原因：

- tokenizer 可以独立比较，不改变 PointNet++ 的局部层级抽象；
- 融合发生在已有 dense feature 上，仍然保留逐点位置；
- tokenizer 只负责产生局部结构表示，原有 GAFM/Transformer 继续负责语言条件 affordance 解码；
- 将来 V2 可以在 `token_feat` 上加语言交互，V4 可以直接使用 `token_meta` 做 2D–3D token 匹配。

### 3.2 连续局部 tokenizer 定义

新增模块建议命名：`model/local_3d_tokenizer.py`。

输入：

- `xyz`: `[B, N, 3]`；
- 可选 `point_feat`: `[B, N, C_in]`。V1 MVP 默认至少使用几何信息，避免依赖 PointNet++ 输出通道变更。

中间量：

- `center_idx`: `[B, M]`，FPS 选出的中心点索引；
- `centers`: `[B, M, 3]`；
- `neighbor_idx`: `[B, M, K]`，每个中心的 KNN 邻居；
- `relative_xyz`: `[B, M, K, 3]`。

推荐计算：

```text
center_idx = FPS(xyz, M)
centers = index_points(xyz, center_idx)
dist = square_distance(centers, xyz)
neighbor_idx = topk(dist, K, largest=False)
relative_xyz = grouped_xyz - centers[:, :, None, :]
local_input = concat(relative_xyz, optional_grouped_point_feat)
local_hidden = LocalMLP(local_input)
token_feat = pool(local_hidden over K)
```

建议的第一版局部编码器：

```text
Linear(3 + C_in → 128)
→ LayerNorm / GELU
→ Linear(128 → 256)
→ GELU
→ max-pooling 或 attention-pooling over K
→ Linear(256 → token_dim=512)
→ LayerNorm
```

不要第一版就使用复杂的局部 Transformer。V1 要优先回答“显式局部 token 是否有效”，而不是扩大模型容量。

### 3.3 KNN 与邻域设计

建议新增 `_knn_points()`，使用现有 `square_distance()`：

```python
# xyz: [B, N, 3], centers: [B, M, 3]
dist = square_distance(centers, xyz)  # [B, M, N]
neighbor_dist, neighbor_idx = dist.topk(k=K, dim=-1, largest=False)
```

理由：

- 与半径搜索相比，KNN 保证每个 token 都有固定 K 个邻居；
- 适合 PIAD/LASO 点数和局部密度不完全一致的情况；
- `neighbor_dist` 可直接用于 token-to-point 插值权重和可视化；
- 若以后要加入空间一致性损失，邻域索引已经保留。

对局部坐标的处理：

```text
relative_xyz = xyz_neighbor - center_xyz
relative_xyz = relative_xyz / (local_scale + eps)
```

`local_scale` 建议使用邻域平均距离或最大邻域距离，避免不同物体尺度造成局部 token 分布漂移。由于数据集已进行 unit-sphere 归一化，第一版也可以先固定不做额外尺度归一化，并把它作为消融项。

### 3.4 token-to-point 映射

这是 V1 的核心约束。不能只生成 `[B, M, C]` 的 token 后再通过全局池化预测 affordance。

推荐使用 token 中心的 3-NN 插值：

```text
对每个原始点 p_j：
  找到距离最近的 r 个 token 中心
  w_ij = exp(-d_ij / τ) / Σ_i exp(-d_ij / τ)
  f_token(p_j) = Σ_i w_ij · z_i
```

张量形式：

- `token_feat`: `[B, C, M]`；
- `point_to_token_idx`: `[B, N, r]`；
- `point_to_token_weight`: `[B, N, r]`；
- `token_point_feat`: `[B, C, N]`。

可复用 `PointNetFeaturePropagation` 的 3-NN 插值逻辑，但建议单独实现 `TokenToPointInterpolator`，原因是：

1. 需要保存 token 中心索引和权重；
2. 需要将 token 元信息提供给后续 V4；
3. 需要方便加入 token dropout、coverage 统计和可视化。

### 3.5 与 PointNet++ 的融合

第一版推荐 gated residual fusion，不直接替换 `fused_feat`：

```text
h = concat(fused_feat, token_point_feat)
h = FusionMLP(h)
g = sigmoid(GateMLP(h))
fused_feat_v1 = fused_feat + γ · g ⊙ h
```

更简化的 MVP：

```text
fused_feat_v1 = fused_feat + γ · Conv1x1(token_point_feat)
```

建议：

- `γ` 为可学习标量，初始化为 `0` 或 `0.1`；
- 门控层最后一层零初始化，使初始模型接近 GEAL baseline；
- 第一版先使用加法或 gated fusion，拼接后大 MLP 作为后续消融；
- `fused_feat_v1` 通道数固定为 512，避免影响现有 decoder、`feature_downsampler`、`gaussian_aff_head` 和 `img_align_proj`。

---

## 4. 文件级改动清单

### 4.1 新增：`model/local_3d_tokenizer.py`

建议包含以下类：

```text
Local3DTokenizer
TokenToPointInterpolator
TokenFusion
```

职责：

- 局部中心采样；
- 固定 KNN 分组；
- 相对位置编码和局部 pooling；
- token-to-point 插值；
- 返回 token metadata。

不建议把训练循环逻辑写入该文件。

### 4.2 修改：`model/branch_3d.py`

#### `__init__`

新增配置读取：

```python
self.tokenizer_cfg = cfg.get("local_tokenizer", {})
self.use_local_tokenizer = self.tokenizer_cfg.get("enabled", False)
```

当 enabled 时创建：

```python
self.local_tokenizer = Local3DTokenizer(...)
self.token_to_point = TokenToPointInterpolator(...)
self.token_fusion = TokenFusion(...)
```

注意：模块应在 `__init__` 中注册，不能在第一次 forward 里惰性创建，否则参数不会进入 optimizer。这一点在本项目之前的 `sam_proj` 实现中已经出现过类似风险。

#### `forward`

建议插入在多尺度融合之后、Transformer decoder 之前。

训练和推理都使用同一 token 路径。只有 debug 模式才额外返回 metadata；默认保持原有返回值数量不变，以减少对 `train_stage2.py`、`evaluation.py`、`evaluation_corrupt.py` 和可视化脚本的影响。

建议接口：

```python
return_aux = cfg.get("return_token_aux", False)

if self.training and return_aux:
    return affordance_map, downsampled_feat, gaussian_aff, patch_feat_3d, fused_feat, token_aux
```

默认 `return_token_aux: false`，不改变现有训练脚本解包逻辑。

### 4.3 修改：`config/train_stage2.yaml`

新增独立配置块，不要直接覆盖现有模型实验：

```yaml
model_3d:
  local_tokenizer:
    enabled: true
    token_dim: 512
    num_tokens: 256
    neighbor_k: 32
    center_sampling: fps
    pooling: max
    interpolate_k: 3
    relative_pos: true
    normalize_local_scale: false
    fusion: gated_residual
    fusion_init: 0.0
    return_token_aux: false
```

建议另存为：

```text
config/train_stage2_v1_tokenizer.yaml
config/evaluation_v1_tokenizer.yaml
config/evaluation_corrupt_v1_tokenizer.yaml
```

不要把 V1 配置直接写进默认 `train_stage2.yaml`，因为旧 checkpoint 与新 tokenizer 参数不兼容，且默认配置当前启用了 IAM/ADM。

### 4.4 修改：`scripts/train_stage2.py`

V1 首轮只需要检查并补充以下内容：

1. `Branch3D` 创建时自动读取 `model_3d.local_tokenizer`。
2. `build_optimizer()` 能够拿到新增 tokenizer/fusion 参数；只要模块在 `Branch3D.__init__` 注册即可，无需单独 `add_param_group`。
3. 保持原有 `HM_Loss` 和现有 2D→3D MSE，不新增 V1 专属损失。
4. 日志增加：
   - `token_num`；
   - `neighbor_k`；
   - tokenizer trainable params；
   - fusion gate 均值/方差；
   - 每 epoch 的 token-to-point 覆盖率和插值距离均值。
5. 训练首轮将 `use_iam_adm: false`，并关闭 `use_new_losses`，避免把 IAM/ADM 或交互图辅助损失误归因给 V1。

如果 tokenizer 只使用 `xyz`，则 `train_one_epoch()` 的 batch 解包不需要改动。

### 4.5 修改：`scripts/evaluation.py` 与 corrupt evaluation

正常评估不需要改变指标计算，但必须保证：

- evaluation config 使用 `training: false` 时，`Branch3D` 仍然创建 tokenizer 和 fusion 模块；
- tokenizer 的结构配置与训练一致；
- checkpoint 加载时不再使用旧的 baseline checkpoint 直接冒充 V1 checkpoint；
- `strict=False` 仅用于兼容检查，不得把 tokenizer 的 missing keys 当成“成功加载”。

建议新增 checkpoint 检查：

```text
若 local_tokenizer.enabled=true：
  检查 checkpoint 是否包含 local_tokenizer.* 和 token_fusion.*
  缺失则直接警告并终止 V1 正式评估
```

### 4.6 新增：区域级评测工具

建议新增：`utils/region_metrics.py` 或 `scripts/evaluate_region_metrics.py`。

至少实现：

- 小/中/大 affordance 区域分桶；
- 区域召回率；
- small-region aIoU；
- boundary F-score；
- 点数下采样曲线；
- token coverage；
- token-to-point 平均插值距离。

当前 `scripts/evaluation.py` 已有 IoU/AUC/SIM/MAE，但这些整体指标不足以直接验证 V1 的局部结构假设。

---

## 5. 配置与实验协议

### 5.1 V1 基准配置建议

为了与当前工程现状兼容，建议从以下设置开始：

```yaml
train:
  seed: 42
  epochs: 50
  batch_size: 4
  use_iam_adm: false
  use_new_losses: false
  render_consistency_weight: 0

model_3d:
  training: true
  emb_dim: 512
  N_p: 64
  fuse_level: true
  local_tokenizer:
    enabled: true
    token_dim: 512
    num_tokens: 256
    neighbor_k: 32
    pooling: max
    interpolate_k: 3
    fusion: gated_residual
    fusion_init: 0.0
```

第一轮不建议直接使用：

- `num_tokens=512`；
- `neighbor_k=64/128`；
- attention pooling；
- 动态 token routing；
- V1 + IAM/ADM；
- V1 + render consistency；
- V1 + 交互图损失。

### 5.2 最小消融矩阵

#### 主消融

| 实验 | PointNet++ | Local tokenizer | Fusion | 目的 |
|---|---:|---:|---|---|
| B0 | ✓ | × | — | 原始基线 |
| B1 | ✓ | ✓ | add | 验证局部 token 是否有效 |
| B2 | ✓ | ✓ | gated residual | 验证门控是否更稳定 |
| B3 | ✓ | ✓ | concat + MLP | 控制融合容量 |

#### tokenizer 消融

| 实验 | M | K | Pooling | 目的 |
|---|---:|---:|---|---|
| T1 | 128 | 32 | max | 低 token 预算 |
| T2 | 256 | 32 | max | 推荐默认 |
| T3 | 512 | 32 | max | 高 token 预算 |
| T4 | 256 | 64 | max | 邻域大小 |
| T5 | 256 | 32 | mean | pooling 对比 |
| T6 | 256 | 32 | attention | 仅在前述配置有效后测试 |

#### 映射消融

| 实验 | token-to-point | 目的 |
|---|---|---|
| P1 | 3-NN inverse-distance | 推荐默认 |
| P2 | 1-NN nearest center | 检查插值平滑作用 |
| P3 | full attention | 仅作为高成本上界，不作为默认 |

### 5.3 控制变量

所有对比必须固定：

- 数据集、Seen/Unseen split；
- 2D checkpoint；
- epoch、batch size、optimizer 和 scheduler；
- PointNet++ 配置；
- 文本编码器与 tokenizer 文本输入；
- 随机种子集合，例如 `42/100/2024`；
- checkpoint 选择标准；
- 推理点数和评估代码。

必须同时报告：

- trainable params；
- 总参数量；
- GPU 显存峰值；
- 单 batch forward 时间；
- 推理 FPS 或每样本延迟；
- token 数和 K；
- 训练是否仍包含 2D teacher 前向。

---

## 6. 训练顺序

### Phase 0：baseline 复现

1. 使用与当前模型结构一致的配置复现 GEAL。
2. 固定并保存 2D teacher checkpoint。
3. 至少跑三个随机种子。
4. 记录 Seen/Unseen、PIAD/LASO 和 clean/corrupt 结果。
5. 补齐 small/mid/large affordance 区域统计。

进入 V1 的条件：baseline 结果可解释，且确认局部区域或点密度是主要瓶颈之一。

### Phase 1：tokenizer smoke test

只验证工程链路，不追求指标：

1. 随机生成 `[B,3,2048]` 点云。
2. 检查 FPS、KNN、token pooling、插值和 fusion 的形状。
3. 检查 `loss.backward()` 后 tokenizer 参数存在非零梯度。
4. 检查没有 NaN/Inf。
5. 检查 `enabled=false` 时输出与 baseline 结构一致。
6. 检查 batch size=1、点数 1024/2048/4096 的边界情况。

### Phase 2：小规模过拟合

1. 选 8–32 个样本。
2. 只使用 `HM_Loss`，不启用其他新损失。
3. 比较 B0 与 B2 能否在小数据上正常下降。
4. 保存 token 中心、邻域和逐点传播结果用于可视化。

### Phase 3：正式训练

1. 先固定 `M=256, K=32, max-pooling, gated residual`。
2. 在 PIAD Seen/Unseen 和 LASO Seen/Unseen 上训练。
3. 每个设置使用三个随机种子。
4. 只在主配置稳定后搜索 M、K、pooling 和 fusion。
5. 最后再测试 corruption，不把 corruption augmentation 混入第一轮主结论。

---

## 7. 验收标准

### 7.1 工程验收

- [ ] `local_tokenizer.enabled=false` 可以运行原 baseline。
- [ ] `local_tokenizer.enabled=true` 可以完成训练、保存和加载 checkpoint。
- [ ] 训练/评估两端结构配置一致。
- [ ] `token_feat` 形状为 `[B, 512, M]`。
- [ ] `token_point_feat` 形状为 `[B, 512, N]`。
- [ ] `fused_feat` 最终仍为 `[B, 512, N]`。
- [ ] `pred_3d` 最终仍为 `[B, N]`。
- [ ] token 中心和邻域索引可在 debug 模式导出。
- [ ] 不引入新的输入模态。
- [ ] GPU 显存和运行时间在可接受范围内。

### 7.2 研究验收

V1 不应只凭整体 IoU 提升判定成功。建议同时满足：

1. PIAD Unseen 或 LASO Unseen 至少一个设置在三个 seed 上有稳定提升；
2. Seen 性能没有明显下降；
3. small affordance 区域召回率或 small-region aIoU 提升；
4. 点数减少时性能下降曲线更平缓，或至少不显著变差；
5. token coverage 不出现大量空间空洞；
6. 结果不能仅由参数量增加解释；
7. gated fusion 的 gate 不应长期饱和为 0 或 1；
8. token 可视化能够显示局部结构，而不是所有 token 都退化为全局平均。

建议的失败判定：

- Seen 提升、Unseen 不提升：可能只是容量增加；
- overall IoU 提升、small-region 指标下降：token 映射或融合损伤边界；
- token usage/coverage 很差：FPS、KNN 或插值实现有问题；
- 三个 seed 方向不一致：先减小 token 分支学习率或使用零初始化 gate；
- 显存明显超预算：优先减小 K，再减小 M，不要先破坏 token-to-point 映射。

---

## 8. 风险与处理策略

| 风险 | 原因 | 处理 |
|---|---|---|
| tokenizer 退化成 PointNet++ 的重复分支 | 融合没有独立局部信息 | 强制使用相对位置，并做 PointNet++/token ablation |
| FPS/KNN 显存高 | `B×M×N` 距离矩阵 | 分块 KNN、先用 M=128/256、K=32 |
| token 覆盖不均 | FPS 对稀疏或非均匀点云敏感 | 记录 coverage 和最近 token 距离；必要时加入随机中心补充 |
| 小区域被插值抹平 | 3-NN 插值过平滑 | 对比 1-NN、3-NN；保留中心点直连 residual |
| 新分支破坏已有效主干 | fusion 权重过大 | gate 零初始化；token 分支较低学习率；保留 residual |
| checkpoint 不兼容 | 新增参数未出现在旧权重 | V1 使用独立实验名；正式评估检查 missing keys |
| 训练与评估结构不一致 | evaluation.yaml 未同步 tokenizer 配置 | 单独维护 V1 train/eval/corrupt 配置 |
| V1 与高级损失混淆 | 当前默认配置启用 IAM/ADM | 首轮明确关闭 IAM/ADM、new losses 和 render consistency |
| token 只提升对象级语义而不提升点级定位 | 丢失 token-to-point 映射 | 把 token 传播后的 dense feature 送入原逐点 decoder，并评估边界指标 |

---

## 9. 推荐实现顺序

### M0：先实现并验证

1. 新建 `model/local_3d_tokenizer.py`。
2. 复用 `farthest_point_sample`、`index_points`、`square_distance`。
3. 实现固定 M、固定 K 的 FPS+KNN+relative position+max-pooling。
4. 实现 3-NN token-to-point 插值。
5. 实现零初始化 gated residual fusion。
6. 在 `Branch3D` 中接到 Step 5 和 Step 6 之间。
7. 只增加 debug 形状和梯度检查，不改 loss。

### M1：跑通训练与加载

1. 新增 `config/train_stage2_v1_tokenizer.yaml`。
2. 新增对应 evaluation/corruption 配置。
3. 关闭 IAM/ADM 与其他新损失。
4. 完成小样本过拟合和 checkpoint round-trip。
5. 确认旧模型 `enabled=false` 可复现。

### M2：正式消融

1. B0/B1/B2/B3 主消融。
2. `M={128,256,512}`。
3. `K={32,64}`。
4. max/mean pooling。
5. 三个随机种子。
6. Seen/Unseen、PIAD/LASO、small/mid/large 区域。

### M3：根据结果决定是否进入 V2/V4

- 若 Unseen 和 small-region 均稳定提升：进入 V2，加入中期语言条件 token 交互。
- 若 token 表示有效但跨模态传递不足：进入 V4，加入 visibility-aware token-level 2D–3D consistency。
- 若只有整体指标小幅提升、局部指标无变化：不要继续堆模块，先检查 token-to-point 映射和评测分桶。
- 若 V1 无提升：停止向 V3/V5 扩展，优先分析 PointNet++ 特征瓶颈、点云密度和 teacher 质量。

---

## 10. 最终推荐的 V1 论文/实验叙事

V1 阶段不要直接声称“建立了可复用的功能词表”或“完成了跨模态 token 对齐”，因为这些属于 V4/V5 尚未验证的内容。

更稳妥的表述是：

> GEAL 原有 PointNet++ 能够形成层级局部特征，但局部结构没有被显式组织为可传播的 token 单元。我们在其旁路引入一个保持中心点—邻域—逐点映射的连续 3D tokenizer，通过相对位置编码聚合局部几何，并以门控残差方式注入原有 dense feature。该设计在不改变 GEAL 2D teacher 和逐点 decoder 的前提下，单独验证显式局部 token 对跨类别 affordance grounding 的作用。

V1 的成功标准不是“模块越多越好”，而是建立一个可复现的因果结论：

```text
相同 PointNet++、相同 2D teacher、相同训练目标
仅增加连续局部 token 表示
→ 是否稳定改善 unseen / small-region affordance grounding？
```

---

## 11. 一句话结论

当前仓库最稳妥的 V1 改法是：**新建独立的 `Local3DTokenizer`，用 FPS+KNN+相对位置编码生成固定数量连续 token，再通过显式 3-NN token-to-point 映射和零初始化 gated residual 融合到 `Branch3D` 的 `fused_feat`，保持 GEAL 现有 2D teacher、GAFM、HM_Loss 和推理接口不变。**