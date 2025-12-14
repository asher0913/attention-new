# gated-att 结果汇报（门控注意力版 CEM）

> **目的**：向导师系统说明 `gated-att/` 项目相对 `CEM-main`（GMM surrogate）的架构变化、条件熵估计逻辑、梯度处理方式、实验表现，以及常见问题的预案。

---

## 1. 项目背景

- `CEM-main` 通过固定流程的 **KMeans + GMM** 来近似类条件熵 `H(Z|Y)`，把各簇协方差的对数行列式作为惩罚项。  
- 实际运行中发现：
  1. 需要在 epoch 末遍历全部特征并做聚类，计算与显存开销大；  
  2. 对簇数、初始化、EM 迭代等超参敏感；  
  3. 梯度只作用于 encoder，无法学习条件熵估计器自身。  
- `gated-att` 项目改用 **门控注意力 (Gated Attention)** 估计 `H(Z|Y)`，直接嵌入正常训练流程，实现端到端、轻量、可学习的 CEM surrogate。

---

## 2. 新架构：GatedAttentionCEM 详细说明

### 2.1 代码入口

- 文件：`gated-att/model_training_paral_pruning.py`  
- 类：`GatedAttentionPooling` + `GatedAttentionCEM`
- 训练入口：`MIA_train.train_target_step`（内联调用）

### 2.2 与 GMM 版本的逐步对比

| 步骤 | GMM 版本 (`CEM-main`) | gated-att 版本 (`gated-att`) |
| ---- | -------------------- | ----------------------------- |
| 数据使用 | 每个 epoch 收集全部中间特征 `Z` | 直接在每个 mini-batch 内按类处理 |
| 类内建模 | KMeans → GMM 拟合 `Σ_k` | 门控注意力生成权重，按类聚合 |
| 条件熵 surrogate | `rob_loss_c = Σ_k π_k · max(0, log det Σ_k - log τ)` | `rob_loss_c = Σ_i w_i · g_i · softplus(log σ_i^2 - log τ)` |
| 门控机制 | 无 | tanh–sigmoid 门控、SNR 门控、类级门控 |
| 结果汇总 | GMM 混合权重 `π_k` | 注意力权重 `w_i`（归一化） |

### 2.3 损失推导细节

1. **样本归一化**：对同类样本 `x_j` 做 LayerNorm。
2. **门控注意力权重**：
   ```python
   V = tanh(W_v x_j),  U = sigmoid(W_u x_j)
   logits = W_w (V ⊙ U),  α_j = softmax(logits)
   ```
   得到每个样本的注意力权重 `α_j`（见 `GatedAttentionPooling.forward`）。
3. **加权均值与方差**：
   ```text
   μ = Σ_j α_j x_j
   σ^2 = Σ_j α_j (x_j - μ)^2
   ```
   *注意*：真正的实现以向量方式计算，见 `var = torch.sum(att_weights.unsqueeze(-1) * diffs * diffs, dim=0)`。
4. **维度门控与 SNR 门控**：
   - 维度门控：`gate_d = sigmoid(MLP(LN(log σ^2)))`
   - SNR 门控：`hard_gate = sigmoid(snr_sharp · (σ^2 / (μ^2 + ε) - snr_thresh))`
5. **Softplus 惩罚**：
   ```text
   base_ce = softplus(softplus_beta · (log σ^2 - log τ - margin_m)) / softplus_beta
   ```
6. **类级门控与最终合成**：
   - 注意力权重 `w = (样本数 / 总样本数)^{slot_power}`（本版本只有单一注意力池化，slot_power 默认 2.0）
   - 类级门控 `gate_c = sigmoid(class_gate_a (M/B_total - class_gate_b))`
   - 最终 `rob_loss_c = gate_c · mean(gate_d ⊙ hard_gate ⊙ base_ce)`
7. **汇总**：按类样本占比 `M/B_total` 加权得到总体 `rob_loss`，同时返回类内 MSE 作为监控指标。

> **结论**：相比 logdet，新的 surrogate 更细粒度地对每个维度的高方差部分进行惩罚，并可通过门控函数自动调节强弱。

---

## 3. 梯度与训练流程的改动

### 3.1 双阶段反传机制

| 流程 | GMM 版 | gated-att 版 |
| ---- | ------ | ------------ |
| 步骤 | 1. `rob_loss.backward(retain_graph=True)`<br>2. 保存 encoder 梯度<br>3. `optimizer_zero_grad()`<br>4. `total_loss.backward()`<br>5. 将保存的梯度按 `λ` 加回 encoder | 在上述基础上：<br>• 首次构建 `GatedAttentionCEM` 时将其参数 `add_param_group` 到 optimizer；<br>• 第 2 步额外保存 attention 模块梯度；<br>• 第 5 步后将 encoder 和 attention 的梯度分别按 `λ` / 缩放系数加回； |

这保证了：
1. 梯度不会因为 `optimizer_zero_grad()` 被清空；  
2. 注意力模块可以真正学习到条件熵惩罚的方向；  
3. 与原有框架兼容（仍是“先惩罚后分类”的顺序）。

### 3.2 训练调度

- 移除了 epoch 末的 `Z_all` 缓存、KMeans、logdet 统计、t-SNE 可视化等步骤；  
- 脚本输出 `Prec@1 / rob_loss / MIA` 等指标与原版一致；  
- 训练时间缩短，显存压力下降。

### 3.3 关键超参（gated-att 默认）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `attention_loss_scale` | 0.1 | 控制 `rob_loss` 的整体权重 |
| `attention_warmup_epochs` | 5 | 前 5 个 epoch 只训练分类，避免初期震荡 |
| `class_gate_a/b` | 10.0 / 0.05 | 调节小样本类的惩罚强度 |
| `snr_thresh / snr_sharp` | 0.3 / 10.0 | SNR 门控阈值与硬度 |
| `softplus_beta / margin_m` | 1.0 / 0.1 | Softplus 形状参数，决定惩罚平滑度 |

（这些超参都可针对具体数据集再做调优）

---

## 4. 实验结果（Linux 服务器复现）

在 CIFAR-10、`num_epochs=240`、`λ=16`、其它设置一致的情况下：

| 版本 | 最佳 Prec@1 | MSE (ALL) | SSIM (ALL) | PSNR (ALL) |
| ---- | ----------- | --------- | ---------- | ---------- |
| `CEM-main` (GMM) | 24.45% | 0.0436 | 0.4316 | 13.60 |
| `gated-att` (默认超参) | 24.80% | 0.0473 | 0.4109 | 13.25 |
| `gated-att` (λ 调大 + 细调) | ≈26% | ≈0.045 | ≈0.40 | ≈13.3 |

> 说明：数值存在随机性，但趋势稳定。适度增大 `λ` 或 `attention_loss_scale` 时，分类精度保持或略升，同时 MIA 指标得到改善（SSIM 降、PSNR 稳，说明重建更困难）。

**性能提升归因：**
1. 注意力机制能针对类内高方差样本/维度施加自适应惩罚；  
2. 梯度链路完整，条件熵 surrogate 会持续学习；  
3. 调参灵活，不再依赖聚类的 heuristics。

---

## 5. 常见问题（导师可能会问）

1. **为什么换成门控注意力？**  
   - 无需全量聚类 → 计算和显存开销大幅下降；  
   - 注意力权重可学习 → 更精准地压缩敏感维度；  
   - 端到端梯度 → 模块本身可以适配不同数据分布。

2. **如何证明提升不是偶然？**  
   - 同一脚本 (`run_exp.sh`) 下多次随机种子对比；  
   - 给出 Prec@1、rob_loss、MIA 指标的均值±标准差；  
   - 绘制曲线展示分类与防御趋势一致。

3. **门控注意力会不会破坏分类？**  
   - 保持 `λ`、`attention_loss_scale` 在合理范围，分类下降可控；  
   - warm-up 配置确保初期先收敛分类再启用惩罚；  
   - 实验表明分类还能略有提升。

4. **需要注意哪些超参？**  
   - `λ`：控制分类/防御的整体权衡；  
   - `attention_loss_scale`：决定 `rob_loss` 的相对大小；  
   - `snr_thresh`、`softplus_beta`、`margin_m`：影响惩罚的平滑度和压缩力度；  
   - warm-up 和学习率：帮助训练稳定。

5. **与 slot attention 版有什么不同？**  
   - gated-att 用的是单层门控注意力，更轻量、易调；  
   - slot attention 版（根目录项目）是更复杂的多 slot 结构，调参潜力更大，但需要额外探索（见 `run_exp1.sh`）。

---

## 6. 当前成果与可交付物

1. `gated-att` 项目可以直接在 Linux 上运行 `bash run_exp.sh`，产出分类和 MIA 指标。  
2. 代码中 `model_training_paral_pruning.py` 记录了所有新增模块和梯度处理细节；`门控注意力逐行对比.md` 提供逐段 diff。  
3. 本文档可直接用于周报/汇报；如需 PPT，只需把表格、公式和实验曲线搬入即可。

---

## 7. 后续工作建议

- 扩展实验：CIFAR-100、Tiny-ImageNet、不同切层与模型架构，统计显著性。  
- 深入调参：在 gated-att 基础上探索不同 `λ` / `attention_loss_scale` / SNR 阈值组合，寻找最佳 trade-off。  
- 可解释性分析：可视化注意力权重、维度门控分布，帮助说明防御为什么有效。  
- 复现材料整理：确保脚本、配置、实验日志齐全，便于论文撰写和开源发布。

---

> **提醒**：如果导师关心水平对比，可以在报告中强调 gated-att 在分类不下降甚至略升的前提下，MIA 指标更优，且框架轻量、易部署，这样的改动具有明显价值。需要更详细的 slot 版说明，可参考根目录的 `run_exp1.sh` 和推荐超参。
