# Conditional Entropy Loss (CEL) Formulations in *CEM-main* and *gated-att*

本文以数学形式抽象描述两个实现中条件熵正则项（Conditional Entropy Loss, CEL）的计算流程，便于在理解原理时脱离具体代码细节。

---

## 1. 基础符号

- $x \in \mathcal{X}$：客户端输入图像；
- $f_{\theta}$：客户端编码器，输出 *smashed data* $z = f_{\theta}(x)$；
- $y \in \{1, \dots, C\}$：样本类别；
- 对 batch $B$，记 $Z_B = \{z_i\}_{i=1}^{|B|}$、$Y_B = \{y_i\}_{i=1}^{|B|}$；
- $\lambda$：CEL 权重；$\sigma^2$：噪声参数；$\tau$：方差阈值；
- $\mathcal{L}_{\text{CE}}$：分类交叉熵损失。

CEL 的目标是让同类 smashed data 在特征空间内保持紧致，从而降低模型反演攻击对客户端输入的可恢复性。

---

## 2. *CEM-main*：基于聚类的 CEL

### 2.1 批次外的统计累计

1. **特征缓存**：在每个 epoch 末，收集所有 batch 的 smashed data $Z_{\text{all}}$ 及对应标签 $Y_{\text{all}}$。
2. **按类聚类**：对每个类别 $c$，将其特征集合 $Z_{\text{all}}^{(c)} = \{ z \mid (z,y)\in (Z_{\text{all}},Y_{\text{all}}), y=c \}$ 进行 $K$-means 或高斯混合聚类：
   \[
   \text{KMeans}\big(Z_{\text{all}}^{(c)}, K\big) \rightarrow \left\{ \mu_{c,k}, \Sigma_{c,k}, \pi_{c,k} \right\}_{k=1}^{K}
   \]
   其中 $\mu_{c,k}$ 是簇中心，$\Sigma_{c,k}$ 是簇协方差近似（简化为对角方差），$\pi_{c,k}$ 是权重（簇占比）。

聚类结果在后续 epoch 中作为离线统计量重复使用。

### 2.2 Batch 级别的 CEL 估计

对当前 batch 中类别 $c$ 的子集 $Z_B^{(c)}$：

1. **最近簇选择**：对每个 $z \in Z_B^{(c)}$，寻找距离最近的簇中心
   \[
   k^\star(z) = \arg\min_k \Vert z - \mu_{c,k} \Vert_2 .
   \]
2. **类内方差估计**：将样本映射到对应簇后，计算
   \[
   v_{c,k} = \frac{1}{|S_{c,k}|}\sum_{z \in S_{c,k}} \Vert z - \mu_{c,k} \Vert_2^2,
   \quad S_{c,k} = \{ z \in Z_B^{(c)} \mid k^\star(z)=k \}.
   \]
3. **加权汇总**（结合离线权重 $\pi_{c,k}$）：
   \[
   \hat{v}_c = \sum_{k=1}^{K} \pi_{c,k}\, v_{c,k}.
   \]

### 2.3 CEL 形式

依据实验设置，CEL 有两种变体：

1. **线性惩罚**：
   \[
   \mathcal{L}_{\text{CEL}}^{\text{lin}} = \sum_{c} \hat{v}_c.
   \]

2. **对数熵惩罚**（log-entropy，可强制方差低于阈值 $\tau$）：
   \[
   \mathcal{L}_{\text{CEL}}^{\log} = \sum_{c} \max\Big( 0, \log(\hat{v}_c + \varepsilon) - \log(\tau + \varepsilon) \Big).
   \]

总损失：
\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{CEL}}.
\]

注意这里只将 $\mathcal{L}_{\text{CEL}}$ 的梯度注入到编码器参数 $\theta$，以直接约束 smashed data。

---

## 3. *gated-att*：基于门控注意力的 CEL

### 3.1 端到端的批内处理

该方法取消了离线统计，直接在当前 batch 上估计类内 dispersion：

1. **特征规范化**：对 batch 中的 smashed data 做 LayerNorm（或零均值规范化）以稳定注意力。
2. **门控注意力**：对每个类别 $c$ 的子集 $Z_B^{(c)}$，构建注意力权重
   \[
   \alpha_{c,i} = \frac{\exp\big( w^\top \big( \tanh(V z_{c,i}) \odot \sigma(U z_{c,i}) \big) \big)}{\sum_{j} \exp\big( w^\top \big( \tanh(V z_{c,j}) \odot \sigma(U z_{c,j}) \big) \big)},
   \]
   其中 $z_{c,i} \in Z_B^{(c)}$，$V,U,w$ 为可训练参数，$\odot$ 表示逐元素乘。

3. **加权统计**：
   - 加权均值：
     \[
     \bar{z}_c = \sum_{i} \alpha_{c,i} z_{c,i};
     \]
   - 加权方差：
     \[
     v_c = \sum_{i} \alpha_{c,i} \Vert z_{c,i} - \bar{z}_c \Vert_2^2.
     \]

4. **批内权重**：使用类别内样本比例
   \[
   \beta_c = \frac{|Z_B^{(c)}|}{|B|}.
   \]

### 3.2 CEL 形式

注意力实现同样支持两种惩罚方式：

1. **线性**：$\mathcal{L}_{\text{CEL}}^{\text{lin}} = \sum_c \beta_c \, v_c$；
2. **对数熵**：
   \[
   \mathcal{L}_{\text{CEL}}^{\log} = \sum_c \beta_c \, \max\Big(0, \log(v_c + \varepsilon) - \log(\tau + \varepsilon) \Big).
   \]

总损失：
\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{CEL}}.
\]

与 *CEM-main* 不同，$\mathcal{L}_{\text{CEL}}$ 的梯度同时更新编码器参数 $\theta$ 和注意力模块参数 $(V, U, w)$，且所有统计均来源于当前 batch，无需额外缓存或聚类。

### 3.3 训练注意点

- 设定 warm-up 轮数 $T_{\text{warm}}$（默认 5），在 $t \leq T_{\text{warm}}$ 时仅训练分类器；
- $t > T_{\text{warm}}$ 后引入 CEL 项并把注意力模块参数加入优化器；
- 通过缩放因子 $\gamma$ 控制 CEL 相对于分类损失的影响：
  \[
  \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \gamma \, \mathcal{L}_{\text{CEL}}, \quad \gamma \in (0,1].
  \]

---

## 4. 对比总结

| 方面 | *CEM-main* | *gated-att* |
| --- | --- | --- |
| 统计方式 | 离线聚类（KMeans/GMM） + 按簇方差 | 批内门控注意力 + 加权方差 |
| 存储需求 | 缓存全量 smashed data | 仅用当前 batch |
| 额外参数 | 无（纯统计） | 注意力网络参数 $(V, U, w)$ |
| 计算触发 | 每几轮重新聚类 | 每个 batch 即时计算 |
| 梯度传播 | 只回传到编码器 | 编码器 + 注意力模块 |
| 超参 | K (簇数)、$\tau$、$\lambda$、log/linear 选择 | warm-up 轮数、注意力隐层维度、$\tau$、$\lambda$、log/linear 选择 |

两种方案都致力于降低 smashed data 的可逆性：前者依赖全局聚类统计模拟条件熵，后者以注意力权重即时近似条件熵，增强了端到端可学习性与效率。

---

## 5. 总损失回顾

无论哪种变体，总目标函数都可以写成
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \Psi\left( \{z_i, y_i\}_{i \in B}; \tau \right),
\]
其中 $\Psi$ 表示对应实现的 CEL surrogate：

- *CEM-main*：$\Psi$ 基于离线聚类并混合簇方差；
- *gated-att*：$\Psi$ 基于门控注意力加权方差。

调节 $\lambda$、$\tau$ 及 surrogate 超参即可迁移论文中的不同实验设置。

---

以上即为两套实现中 CEL 的数学抽象与核心差异。若后续需要拓展，可考虑将门控注意力与聚类统计结合，或引入其他衡量类内紧致度的可微 surrogate。
