# 🧠 TumorScope-FlexConv

**A Lightweight 3D Medical Image Segmentation System for Tumor Detection**

---

## 📌 项目简介

本项目基于深度学习医学影像分割技术，提出了一种改进模型 **TumorScope-FlexConv**，用于解决复杂医学图像（特别是3D CT数据）中的肿瘤分割问题。

在经典模型 MedNeXt 的基础上进行优化，实现：

* 更高的分割精度
* 更低的计算成本
* 更好的泛化能力

同时，项目还开发了一个**图形化系统（GUI）**，支持三维可视化和交互操作，提升实际应用价值。 
<img width="1306" height="761" alt="image" src="https://github.com/user-attachments/assets/b065cfae-1582-46f3-acaf-7508e5b6ffbd" />

---

## 🚀 项目亮点（Highlights）

* ✅ 基于 **3D医学图像分割**
* ✅ 自研模型 **TumorScope-FlexConv**
* ✅ 引入 **残差增强注意力机制（REAM）**
* ✅ 支持多数据集（KiTS / BraTS / MSD等）
* ✅ 模型轻量化（参数减少约30%）
* ✅ 提供 **3D可视化 + GUI界面**
* ✅ 支持非专业用户操作

---

## 🧩 技术栈（Tech Stack）

* Python
* PyTorch
* NumPy / SciPy
* PyQt5（GUI界面）
* NIfTI 医学影像处理
* 深度学习（CNN / Attention机制）

---

## 🏗️ 模型设计（Model Architecture）

本项目以 **MedNeXt** 为基准模型，提出改进结构：

### 🔹 核心改进

* 倒置瓶颈结构（Inspired by Transformer）
* 深度可分离卷积（降低计算量）
* 多尺度特征融合
* 残差连接（提升训练稳定性）
* 深监督机制（加快收敛）

### 🔹 注意力机制（REAM）

* 通道注意力 + 空间注意力
* 残差增强信息流
* 提升关键区域识别能力

👉 有效提升模型对复杂结构（如胰腺、结肠肿瘤）的分割能力 

---

## 📊 数据集（Datasets）

使用多个公开3D医学数据集：

* KiTS（肾脏肿瘤）
* BraTS（脑肿瘤）
* MSD_Lung（肺）
* Liver（肝脏）
* Pancreas（胰腺）
* Colon（结肠）

👉 针对小目标 & 复杂结构任务进行了重点优化 

---

## ⚙️ 训练细节（Training Details）

* 数据预处理：

  * Crop（裁剪非零区域）
  * Resample（统一空间分辨率）
  * Normalization（标准化）

* 数据增强：

  * 随机翻转 / 旋转
  * 强度变换
  * 噪声注入
  * 弹性形变

* 损失函数：

  * Dice + CrossEntropy（DiceCELoss）

* 验证方式：

  * 五折交叉验证

---

## 📈 评价指标（Metrics）

* Dice Score
* IoU
* HD95
* Sensitivity / Specificity
* RVE（体积误差）

👉 在多个数据集上相比基线模型有明显提升 

---

## 🖥️ 系统展示（System Demo）

### 功能：

* 📂 上传医学影像（NIfTI）
* 🧠 自动肿瘤分割
* 📊 三维可视化
* 🔍 多视角查看（轴向 / 冠状 / 矢状）
* 🎛️ 掩膜透明度调节
* 🎨 标签与颜色自定义

👉 支持医生进行辅助诊断与分析 

---

## 📦 项目结构（Project Structure）

```bash
TumorScope-FlexConv/
├── models/          # 模型结构
├── data/            # 数据集
├── utils/           # 工具函数
├── gui/             # PyQt界面
├── train.py         # 训练脚本
├── infer.py         # 推理脚本
├── requirements.txt
└── README.md
```

---

## ▶️ 使用方法（How to Run）

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 模型训练

```bash
python train.py
```

### 3️⃣ 推理测试

```bash
python infer.py
```

### 4️⃣ 启动GUI

```bash
python app.py
```

---

## 🌍 应用场景（Applications）

* 🏥 医学影像辅助诊断
* 🧬 肿瘤检测与分析
* 📚 医学教育与训练
* 📊 医疗数据分析

---

## 📌 项目成果

* 提出轻量化医学分割模型
* 在困难任务（胰腺/结肠）上性能显著提升
* 构建完整工程系统（模型 + GUI）
* 具备实际落地潜力

---

## 👨‍💻 作者（Author）

* Huang Zhen
* SliceMasters Team
* Chongqing University of Arts and Sciences

---

## ⭐️ 如果这个项目对你有帮助，欢迎 Star！

---

## ✅ 接下来建议你做

我帮你说句实话（对你提升很关键）👇

👉 这个 README 已经**能打面试了，但还能再升级：**

* 加几张 **分割效果图（非常重要🔥）**
* 加一个 **模型对比图（PPT里有）**
* 如果可以 → 上传一个 demo 视频

---

如果你愿意，我可以帮你再做一步升级👇
👉 帮你改成 **“面试官一看就想问你问题”的顶级GitHub项目”**（含截图+排版+亮点强化）
