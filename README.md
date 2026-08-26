# CausalMVC BDGP Clean

这是当前稳定 BDGP 模型的单入口整理版，不依赖 `v2/v3/fixed/stable_final` 等历史脚本。

## 文件职责

| 文件 | 职责 |
|---|---|
| `train.py` | 唯一训练入口、20+10 两阶段训练、最终评估与保存 |
| `model.py` | 编码器、Common/Specific 分解、解码器、GRL、共享原型和跨视图预测器 |
| `config.py` | 少量常用实验参数及命令行解析 |
| `unit.py` | 结构教师、损失函数、EMA 原型更新、聚类指标 |
| `data_load.py` | BDGP 加载、L2 归一化和无标签缺失 mask |

## 训练流程

1. 从观测数据构建固定结构教师与伪标签。
2. 前 20 epoch：仅使用 Masked Reconstruction，Common 和 Specific 共同参与重构。
3. 后 10 epoch：联合优化 Reconstruction、GRL、Pair、PAM、Structure 和 Cross-view Loss。
4. 使用 Cross Predictor 补全缺失视图的 128 维潜表示。
5. 对两个补全后的 Common 表示求均值、归一化并执行 KMeans。

当前版本已经移除最终稳定日程中从未触发的 Orthogonality、Cluster Head CE 和 Balance 分支。

## Terminal 运行

```powershell
cd D:\Data_Mining\Code\Experiment\CausalMVC\bdgp_clean
powershell -ExecutionPolicy Bypass -File .\run_train.ps1
```

指定训练种子和独立 mask 种子：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train.ps1 -Seed 61 -MaskSeed 61
```

指定数据和输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train.ps1 `
  -Data "D:\Data_Mining\Code\Datasets\BDGP\BDGP.mat" `
  -Output "outputs\bdgp_clean"
```

`run_train.ps1` 复用当前项目已经验证的 `lycenv` 依赖环境；如果你的终端已经正确激活该环境，也可以直接执行 `python train.py`。

常规实验只暴露 `data/output/seed/mask-seed/missing-rate` 五个命令行参数。网络维度、损失权重和教师细节属于已验证稳定方案，分别固定在实际使用它们的模块附近。

## 输出

- `outputs/bdgp_clean/seed_61.json`：配置、逐 epoch 损失、教师诊断及最终 ACC/NMI/ARI/PUR。
- `outputs/bdgp_clean/seed_61.pt`：模型参数、输入维度和配置。

真实标签不会进入结构教师、损失函数或 checkpoint 选择，只用于最终指标报告。
