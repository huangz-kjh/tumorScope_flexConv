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
<img width="1600" height="798" alt="image" src="https://github.com/user-attachments/assets/dcf7757a-df04-470c-ad1a-4ef245430eb0" />
<img width="1119" height="789" alt="image" src="https://github.com/user-attachments/assets/2a522b84-2c73-4ea2-8bb2-2543c49679e0" />


---

## 📊 数据集（Datasets）

使用多个公开3D医学数据集：

* KiTS（肾脏肿瘤）
* BraTS（脑肿瘤）
* MSD_Lung（肺）
* Liver（肝脏）
* Pancreas（胰腺）
* Colon（结肠）
<img width="1018" height="771" alt="image" src="https://github.com/user-attachments/assets/0d044b3b-4442-4aff-ab56-d187cb774b81" />

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
<img width="1092" height="607" alt="image" src="https://github.com/user-attachments/assets/ceb09195-3f98-4286-8f47-b8a3a9b09c62" />


---

## 🖥️ 系统展示（System Demo）

### 功能：

* 📂 上传医学影像（NIfTI）
* 🧠 自动肿瘤分割
* 📊 三维可视化
* 🔍 多视角查看（轴向 / 冠状 / 矢状）
* 🎛️ 掩膜透明度调节
* 🎨 标签与颜色自定义
<img width="1036" height="631" alt="image" src="https://github.com/user-attachments/assets/7a883575-8a1b-4246-9c9c-4d127b6fdb88" />
<img width="957" height="638" alt="image" src="https://github.com/user-attachments/assets/545bfbd3-d450-47f8-9c8b-830237999da8" />
<img width="1194" height="488" alt="image" src="https://github.com/user-attachments/assets/d400af1d-e9d8-420f-8dc8-728001efc43b" />
<img width="973" height="690" alt="image" src="https://github.com/user-attachments/assets/1a382ad9-583c-4155-ba50-41e162972b39" />

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
