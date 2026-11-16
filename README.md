# Sea Clutter Target Detection with GraphSAGE

《面向时变海杂波的智能目标检测方法研究》的可运行复现项目。使用 MATLAB 将 IPIX 雷达的原始 NetCDF 回波转换为时空图，再用 PyTorch 对目标距离门的回波窗口进行二分类。仓库包括数据生成、五组模型变体和低虚警率评估。

## 数据

从 [McMaster IPIX Dartmouth 数据页面](https://soma.mcmaster.ca/ipix/dartmouth/datasets.html) 下载论文表 4.1 对应的 `#17、#26、#30、#31、#40、#54、#280、#310、#311、#320`，按原文件名解压到 `data/raw/`。数据使用条件见该页面。每条记录有 131,072 个脉冲、14 个距离门和 HH、HV、VH、VV 四种极化。

以距离门为单位对 I/Q 分窗，每个 64 脉冲窗口生成一个包含 64 个 FFT 幅度值的节点。节点按 8 邻域连接，相邻节点的谱幅度余弦相似度作为边权。每张图有 28,672 个节点和 217,008 条有向边。主目标距离门标记为 1，保护距离门仅参与消息传播。其余距离门作为海杂波，每个参与监督的距离门按固定种子独立抽取约 70% 的窗口训练，其余窗口用于测试。

## 快速开始

在项目根目录运行以下命令。已在 MATLAB R2025a、Python 3.12、PyTorch 2.5.1 的 Windows CPU 环境验证。生成全部数据需要约 315 MB 磁盘空间。

```bash
python -m pip install -r requirements.txt
matlab -batch "addpath('matlab'); build_ipix_graphs('data/raw','data/graphs',26,'hh');"
python train.py --graphs data/graphs --ids 26 --polarizations hh --epochs 3 --device cpu --output runs/smoke.csv
```

全量生成并训练。每个记录与极化分别训练五个变体，每个配置使用 300 轮 Adam。已有 MATLAB 产物时从训练命令开始即可。

```bash
matlab -batch "addpath('matlab'); build_ipix_graphs('data/raw','data/graphs');"
python train.py --graphs data/graphs --output runs/paper.csv
python summarize.py runs/paper.csv
```

`--ids 26 54 --polarizations hh` 可重跑仓库中的两条 HH 实验，`--variants E3 E4` 可聚焦全局分支，`--gate-bias -2` 至 `--gate-bias 2` 可比较不同初始偏置。若 Windows Conda 环境在导入 SciPy 和 PyTorch 时报告 OpenMP 运行时冲突，可使用独立 Python 虚拟环境。CUDA 运行需要匹配显卡架构的 PyTorch 构建，可用 `--device cpu` 强制走 CPU。

## 模型与评估

| 变体 | 节点度 | 余弦边权 | PageRank + Gate |
| --- | --- | --- | --- |
| E0 | | | |
| E1 | ✓ | | |
| E2 | | ✓ | |
| E3 | ✓ | ✓ | |
| E4 | ✓ | ✓ | ✓ |

E0–E3 对应论文表 4.3 和原 `GCN.py` 中仍在运行的结构消融。保留三层邻居加权求和、自身线性变换、BatchNorm、Dropout 和加权交叉熵。邻居求和用稀疏矩阵乘法实现，全批次计算。

E4 根据论文第 2.6–2.7 节和图 3.2 补写。先在八邻域图上计算未加权 PageRank，再以 PageRank 对节点表示加权汇聚为隐式全局节点。Gate 在第二层表示和第三层输出处融合局部与全局信息。全局投影、隐式全局节点和门控位置是本仓库的具体实现选择。原图数据及原 Gate 源码缺失，边权采用旧 MATLAB 注释中的频谱余弦相似度。MATLAB 导出为 1 基索引，Python 加载时转换为 0 基索引。随机种子为重新设定的 42，训练类别权重按训练掩码统计。

- 默认 `--protocol paper` 按论文第 3.2.4 节，使用测试集杂波分数确定经验 FAR ≤ 0.001 时的阈值，再报告同一测试集的 PD。阈值由分数顺序统计量给出，判决使用严格大于阈值，替代旧脚本的两轮网格扫描。
- `--protocol validation` 从原训练掩码分出一组校准节点，以其分数定阈值，在剩余的原测试节点上报告实测 FAR 和 PD。测试 FAR 可能高于校准目标值。
- 两种配置都在同一记录的图上进行传导式评估，训练阶段的消息传播及 BatchNorm 可接触测试节点特征。这里的 PD 描述同记录表现，跨记录泛化需要按采集文件留出并重新训练。

## 实测记录

[`results/hh.csv`](results/hh.csv) 保存真实 IPIX 数据上 `#26/HH` 与 `#54/HH` 的 300 轮复跑。下表使用 `paper` 协议，每行共用同一记录与掩码，PD 按百分比显示。

| 记录 | E0 | E1 | E2 | E3 | E4 | 实测 FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| #26 HH | 42.67% | 33.55% | 42.67% | 36.48% | 50.65% | 0.0888% |
| #54 HH | 99.02% | 99.35% | 99.02% | 99.02% | 99.67% | 0.0977% |

E4 在以上两个记录中优于 E3，节点度和余弦边权单独使用时的效果随记录变化。论文第 4 章使用 10 条记录与四种极化做汇总，完整训练矩阵和论文表格的数值复核仍待完成。具体差异和历史脚本衔接见 [复现记录](REPRODUCTION.md)。

[`results/hh_validation.csv`](results/hh_validation.csv) 给出 #26/HH 的独立阈值校准结果。E4 的测试 PD 为 50.81%，FAR 为 0.1333%（9/6,754），超过 0.1% 的校准目标。它记录了阈值从校准节点移到测试节点后的实测变化。
