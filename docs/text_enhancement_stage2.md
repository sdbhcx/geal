# 设计:Stage 2 文本增强 — LLM 模板扩充 + 功能性描述注入

> 目标:通过**纯数据增强**手段,在**不修改模型结构**的前提下,提升 GEAL Stage 2 的文本利用质量。两个正交方向:(A) LLM 生成更多样化的 affordance 问题模板;(B) 为每个问题拼接功能性定义描述,拉开 CLIP/RoBERTa 特征空间中不同 affordance 的语义距离。
> 关联代码:`dataset/piad.py` + `dataset/laso.py`(数据加载)、`scripts/generate_augmented_questions.py` + `scripts/generate_functional_descriptions.py`(离线生成)、`config/train_stage2.yaml`(配置开关)。

---

## 0. 关键前提发现

1. **不修改模型结构**。当前 `Branch2D`/`Branch3D` 的文本编码器(CLIP/RoBERTa)、GAFM 融合、Transformer Decoder 完全不动。改动只发生在数据加载层(`_sample_question` 和 `__getitem__`),且**训练期和推理期行为一致**(测试时也走增强文本,因为推理时也依赖文本输入区分 affordance)。
2. **两个方向正交,可独立开关**。A(模板扩充)和 B(功能性描述)互不依赖,可叠加,也可单独验证。
3. **离线生成,在线零开销**。LLM 推理只在预处理阶段跑一次,训练时只读 CSV,不增加前向时间。
4. **已有 15 个模板是高质量种子**。`Affordance-Question.csv` 的 15 个 rephrasing 是人工编写的,质量高但数量少。LLM 生成时以它们为 few-shot 示例,可保证生成质量。

---

## 1. 当前文本流的瓶颈分析

### 1.1 模板数量有限

```
当前: 每个 (object, affordance) 对 → 15 个模板 (Question0-14)
训练时随机采样 1/15 → 整个 epoch 每个样本只见过 15 种问法
```

对于 23 个 object × 18 个 affordance = 414 个有效对,总共只有 414×15 = 6210 条文本。对于一个 CVPR 级别的模型,这个量级太单薄。

### 1.2 短文本的 CLIP 特征区分度不足

当前问题格式:
```
"Where is the part for sitting?"
"Where is the part for supporting?"
"Where is the part for laying?"
```

CLIP 文本编码器对这类**单动词替换**的短句,输出的特征向量在余弦空间区分度有限。`sit` / `support` / `lay` 在 CLIP 嵌入空间中距离较近,导致 3D 分支的 GAFM 融合时难以区分相近 affordance。

### 1.3 缺乏上下文语义

现有问题是"指向性"的(asking WHERE),但没有"描述性"的(describing WHAT)。如果模型在语义上理解"坐"需要"平坦、水平、承重的表面",它对 `sit` 区域的定位会更准确。

---

## 2. 方案 A:LLM 驱动的问题模板扩充

### 2.1 数据流

```
离线阶段:
  Affordance-Question.csv (15 cols/row)
       │
       ├── LLM 生成 (Qwen2.5-7B-Instruct)
       │   ├── 输入: 原始 5 个模板作为 few-shot 示例
       │   ├── 生成: 50 个新问题
       │   └── 去重 + 过滤 (与已有问题比较,去掉重复/低质)
       │
       └── Affordance-Question-Augmented.csv (15+50=65 cols/row)
            Question0..14: 原始模板
            Question15..64: LLM 生成

在线阶段 (PiadDataset.__getitem__):
  _sample_question() 从 65 列中随机选一列
       │
       └── 拼接视角前缀: f"This is a depth map of a {class} viewed {vp}. {question}"
```

### 2.2 LLM Prompt 设计

```
System:
You are a data augmentation assistant for 3D affordance detection.
Generate diverse questions asking about which part of an object enables
a specific function. Guidelines:
1. Vary sentence structure: "Where is", "Which part", "Show me", "Identify", "Point to"
2. Vary vocabulary: use synonyms for the affordance action
3. Cover broad ("the sitting area") and precise ("the seat cushion")
4. Do NOT include object name (added by template later)
5. Each must end with "?" and be self-contained

User:
Generate {n} diverse questions for affordance "{affordance}" on "{object}".
Existing examples for reference:
{existing_5_questions}

Generate {n} NEW questions different from above:
```

### 2.3 去重策略

```python
def deduplicate(new_questions, existing_questions, threshold=0.85):
    """用 CLIP 文本编码器计算语义相似度,去掉与已有问题过相似的。"""
    all_questions = existing_questions + new_questions
    embeddings = clip_text_encoder.encode(all_questions)
    # 计算新问题与所有已有问题的最大余弦相似度
    for i, q in enumerate(new_questions):
        sim = cosine_sim(embeddings[len(existing_questions)+i], 
                        embeddings[:len(existing_questions)])
        if max(sim) > threshold:
            new_questions.remove(q)  # 语义重复,丢弃
    return new_questions
```

### 2.4 与现有系统的兼容性

- 增强版 CSV 多 50 列,列名 `Question15`-`Question64`,与原始格式完全一致
- `_sample_question()` 只需修改 `all_question_cols` 的列表长度
- 测试集仍固定用 `Question0`,不受影响
- 如果增强版 CSV 不存在,自动回退原始版,保持向后兼容

---

## 3. 方案 B:功能性描述注入

### 3.1 核心思路

原始问题:
```
"Where is the part for sitting?"
```

拼接功能性描述后:
```
"Sitting involves resting one's body weight on a flat, horizontal surface
that is elevated from the ground. The seat must be broad enough to support
the buttocks and thighs. Where is the part for sitting?"
```

**为什么有效**:当 CLIP 编码器处理包含功能性定义的文本时,输出的 `[B, 40, 512]` 特征向量中,与 affordance 相关的语义维度被增强,不同 affordance 的特征在空间中的夹角增大 → Transformer Decoder 的 `einsum` 注意力计算更准确。

### 3.2 数据流

```
离线阶段:
  Affordance-Question.csv
       │
       ├── LLM 生成 (Qwen2.5-7B-Instruct)
       │   ├── 输入: (object, affordance) 对
       │   ├── Prompt: "Write a 1-2 sentence functional definition of
       │   │   {affordance} on {object}. Focus on physical interaction
       │   │   and required part properties."
       │   └── 输出: 功能性描述文本
       │
       └── Affordance-Functional-Desc.csv
            Object, Affordance, FunctionalDesc
            Chair,  sit,        "Sitting involves resting..."

在线阶段 (PiadDataset.__getitem__):
  _sample_question() → question
       │
       ├── [可选] 拼接功能性描述
       │   策略: f"{func_desc} {question}"  (prefix 模式,默认)
       │   备选: f"{question} {func_desc}"  (suffix 模式)
       │
       └── 拼接视角前缀 → 最终文本
```

### 3.3 LLM Prompt 设计

```
System:
You are an expert in functional object understanding. For each (object, 
affordance) pair, write a concise functional definition (1-2 sentences)
describing the physical interaction and the properties of the object part
that enable this function. Be precise and avoid generic statements.

User:
Write a functional definition for affordance "{affordance}" on "{object}".

Example for (chair, sit):
"Sitting involves resting one's body weight on a flat, horizontal surface
that is elevated from the ground. The seat must be broad enough to support
the buttocks and thighs, and positioned at a height allowing feet to rest
on the ground."

Now write for ({object}, {affordance}):
```

### 3.4 拼接策略对比

| 策略 | 格式 | 优势 | 劣势 |
|------|------|------|------|
| **prefix** (默认) | `"{func_desc} {question}"` | 描述先进入编码器,影响全局语义 | 描述过长时可能稀释问题焦点 |
| **suffix** | `"{question} {func_desc}"` | 问题先进入,保留原始指向性 | 描述可能被截断(max_length=40) |
| **separate** | `"{question}"` + 描述作为额外 token | 控制变量清晰 | 需修改 encoder 输入 |

**推荐**:prefix 模式。CLIP 的 attention 是双向的,前缀能影响所有 token 的编码;且 40 个 token 的限制下,一个 1-2 句的描述(约 20-30 tokens)加一个短问题(约 10 tokens)刚好放得下。

---

## 4. 组合后的完整数据流

```
Affordance-Question-Augmented.csv (65 cols)
  └─ _sample_question() 随机选 Question{0..64}
       └─ question: "Where is the part for sitting?"
            │
            ├─ [B] 查找 Affordance-Functional-Desc.csv
            │    └─ func_desc: "Sitting involves resting..."
            │         │
            │         └─ 拼接: f"{func_desc} {question}"
            │              → "Sitting involves resting... Where is the part for sitting?"
            │
            └─ 拼接视角前缀:
                 f"This is a depth map of a {class} viewed {vp}. {final_text}"
                 → "This is a depth map of a chair viewed from the front view.
                    Sitting involves resting... Where is the part for sitting?"
                 │
                 ▼
            CLIP/RoBERTa 编码 → [B, 40, 512] (与原始结构完全一致)
```

---

## 5. 配置变更

### config/train_stage2.yaml

```yaml
dataset:
  category: piad
  ...
  # === 方案 A: LLM 模板扩充 ===
  use_augmented: true           # 是否使用增强版 CSV
  n_augmented_questions: 50     # 新增列数 (Question15..Question64)
  
  # === 方案 B: 功能性描述 ===
  use_functional_desc: true     # 是否拼接功能性描述
  func_desc_strategy: "prefix"  # prefix | suffix
```

### config/train_stage1.yaml

同样新增上述字段,保持两阶段配置一致。

---

## 6. 代码落点

### 6.1 新增文件

| 文件 | 内容 | 规模 |
|------|------|------|
| `scripts/generate_augmented_questions.py` | LLM 离线生成多样化问题 | ~150 行 |
| `scripts/generate_functional_descriptions.py` | LLM 离线生成功能性描述 | ~120 行 |

### 6.2 修改文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `dataset/piad.py` | `__init__` 新增参数/加载增强 CSV/加载功能描述;`_sample_question` 扩充采样池;`__getitem__` 拼接功能描述 | ~40 行 |
| `dataset/laso.py` | 同上(保持两数据集接口一致) | ~30 行 |
| `config/train_stage2.yaml` | `dataset` 段新增 4 个字段 | ~5 行 |
| `config/train_stage1.yaml` | 同上 | ~5 行 |

### 6.3 不修改的文件

| 文件 | 原因 |
|------|------|
| `model/branch_2d.py` | 纯数据增强,不涉及模型结构 |
| `model/branch_3d.py` | 同上 |
| `model/fusion_block.py` | 同上 |
| `utils/clip_text_encoder.py` | 同上 |
| `utils/loss.py` | 同上 |
| `scripts/train_stage2.py` | 参数通过 config 传递,dataset 内部处理 |

---

## 7. 设计取舍与风险

### 7.1 为什么不做 prompt 集成(Prompt Ensemble)

一种替代方案是:训练时同时输入 15 个问题,取 logits 均值。但:
- 需要修改模型 forward 接受多文本
- 推理时也要做 ensemble,增加 15 倍计算量
- 不如让单个文本本身更强

### 7.2 为什么不用 LLM 在线生成

另一种方案是训练时实时调 LLM 生成文本:
- 延迟不可控(每个 batch 都要等 LLM)
- 成本高(每 epoch 生成 400×12=4800 次)
- 离线生成一次,质量可控且零训练开销

### 7.3 功能性描述过长的风险

CLIP 的 `max_length=40`,如果描述太长会被截断:
- 原始问题约 10 tokens
- 视角前缀约 15 tokens
- 功能性描述应控制在 15 tokens 以内 → 1-2 短句

**缓解**:在生成 prompt 中明确要求"1-2 short sentences, under 15 words"。

### 7.4 增强问题质量不可控

LLM 可能生成语法错误、语义偏离或无意义的问题:
- **缓解 1**:以原始 5 个高质量模板作为 few-shot 示例,约束生成分布
- **缓解 2**:用 `Question0` 作为回退(如果抽到的新列是空值)
- **缓解 3**:生成后人工抽检(每 50 个抽检 1 个,批量标注)

---

## 8. 落地顺序建议

1. **先跑方案 B(功能性描述)**:改动最小(只改 `__getitem__` 中一句话拼接),收益可立即在验证集上看到
2. **再跑方案 A(模板扩充)**:需要离线生成 + CSV 整合,但数据量收益更大
3. **最后叠加 A+B**:两个方向正交,理论上效果叠加

---

## 9. 验证方案

### 9.1 离线验证(生成后)

```bash
# 检查生成 CSV 完整性
python -c "
import pandas as pd
df = pd.read_csv('Affordance-Question-Augmented.csv')
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(f'新增列: {[c for c in df.columns if c.startswith(\"Question\") and int(c[8:]) >= 15][:3]}')
# 检查空值率
null_rate = df.iloc[:, 15:].isnull().sum().sum() / (df.shape[0] * 50)
print(f'New columns null rate: {null_rate:.2%}')
"
```

### 9.2 特征空间分析

```python
# 对比增强前后 CLIP 嵌入空间的类间/类内距离
# 预期: 功能性描述 → 类间距离 ↑, 类内距离 ↓
```

### 9.3 训练验证

4 组对比实验(固定 seed,各跑 1 epoch 看 loss 收敛):

| 配置 | use_augmented | use_functional_desc | 预期 |
|------|:---:|:---:|------|
| baseline | false | false | 当前基线 |
| A only | true | false | 文本多样性提升 |
| B only | false | true | 语义区分度提升 |
| A+B | true | true | 两者叠加 |

每组记录:训练 loss 曲线、验证集 IOU/AUC/SIM/MAE。