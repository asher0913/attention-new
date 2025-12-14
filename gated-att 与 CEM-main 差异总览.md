# gated-att 与 CEM-main 差异总览

## 1. 运行层面的差异
- `gated-att/run_exp.sh:2` 重新启用 `cd "$(dirname "$0")"`，确保在 Linux 上直接执行脚本即可进入 `gated-att` 目录运行，避免误用其它工程。
- 训练/攻击脚本 `main_MIA.py`、`main_test_MIA.py` 仅保留 `model_training_paral_pruning`，彻底移除旧的 `model_training` 入口。
- 文件清理：删除 `GMM.py`、`model_training_GMM.py`、`model_training.py`，禁止回退到高斯混合实现。

## 2. 核心代码差异：`model_training_paral_pruning.py`

### 2.1 新增门控注意力模块
- **新增类** `GatedAttentionPooling` 与 `GatedAttentionCEM`（`gated-att/model_training_paral_pruning.py:35-116`）：对应 Ilse 等人（2018）“gated attention MIL” 结构，用以构造条件熵代理。
- **属性初始化** `self.gated_attention_cem = None`（`gated-att/model_training_paral_pruning.py:435`），取代原有的 `use_attention_cem`、`centroids_list` 等缓存。

### 2.2 训练步骤改写
- `train_target_step` 签名精简为 `(..., random_ini_centers, client_id)`（`gated-att/model_training_paral_pruning.py:744`），不再传递各种 centroid/variance 列表。
- 在批量维度上直接拉平特征，调用 `GatedAttentionCEM`（`gated-att/model_training_paral_pruning.py:748-778`）：
  - 若特征出现 NaN/Inf，直接跳过本批次的鲁棒损失。
  - 首次使用时按特征维度自适应设定隐藏层宽度。
- 其他正则（加噪、DP、Dropout、GAN 等）以及“双重反传”流程保持不变：先对 `rob_loss` 反传缓存梯度，再累加分类损失梯度（`gated-att/model_training_paral_pruning.py:799-839`）。

### 2.3 删除的 GMM/KMeans 逻辑
- 移除以下函数与相关调用：
  - `kmeans_plusplus_init`
  - `kmeans_cuda`
  - `apply_gmm_with_pca_and_inverse_transform`
  - `compute_class_means`
- 删除整个 epoch 尾部的“全量特征缓存 + 聚类 + determinant 日志保存 + t-SNE 可视化”等流程（原始 CEM-main 在 `epoch` 循环中大量调用 `Z_all`/`centroids_list` 的片段均已去除，现仅剩占位注释）。
- 清理所有 `centroids_list/cluster_variances_list/weights_list` 变量以及对应日志输出。

### 2.4 其他小改动
- 顶部 import 去掉 `GMM.fit_gmm_torch`、`GaussianMixture`、`KMeans`、`PCA` 等依赖。
- 保留注释说明“gated attention 替换 KMeans/GMM”以提醒未来维护者。

## 3. 辅助脚本改动

| 文件                             | 修改                                  | 目的                                         |
| -------------------------------- | ------------------------------------- | -------------------------------------------- |
| `gated-att/main_MIA.py:6`        | `import model_training_paral_pruning` | 避免引用被删除的 GMM 版本                    |
| `gated-att/main_test_MIA.py:9`   | 同上                                  | 同上                                         |
| `gated-att/run_exp.sh:2`         | 去掉注释直接 `cd`                     | 保证脚本从正确目录启动                       |
| `gated-att/GATED_ATT_CHANGES.md` | 新增文档                              | 记录迁移背景与变更摘要（中文版可参考本说明） |

## 4. 目录结构差异
- `gated-att/` 不再包含：
  - `GMM.py`
  - `model_training_GMM.py`
  - `model_training.py`
- 新增 `GATED_ATT_CHANGES.md`（当前中文说明即可替代或更新）。

---

综上，`gated-att` 相比 `CEM-main` 的唯一区别就是用门控注意力直接取代原有的 GMM/KMeans 条件熵代理，其余训练、防御、记录、攻击脚本保持原有结构。只要在 Linux 上进入 `gated-att` 并执行 `bash run_exp.sh`，即可对比“原版 GMM”与“门控注意力”之间的效果差异。若需要与 `CEM-main` 直接对照，可在同样环境下运行两份项目的 `run_exp.sh`，对比输出日志或模型表现。