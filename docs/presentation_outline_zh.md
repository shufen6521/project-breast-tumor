# 乳腺肿瘤图像识别 Presentation 提纲

## 1. 研究背景

- 乳腺肿瘤早筛对临床诊断有重要意义。
- 超声图像成本较低、应用广泛，但人工判读依赖经验。
- 本项目使用深度学习辅助识别乳腺超声图像中的 `normal`、`benign`、`malignant` 三类。

## 2. 数据集

- 数据集：BUSI Breast Ultrasound Images Dataset。
- 类别：正常、良性、恶性。
- 数据特点：图像数量适中，带有医学图像场景，适合课程训练和展示。
- 参考来源：
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/
  - https://huggingface.co/datasets/Angelou0516/BUSI

## 3. 方法

- 使用迁移学习构建基座模型。
- 对比 `ResNet18` 和 `EfficientNet-B0`。
- 输入图像统一缩放到 `224x224`。
- 使用水平翻转、旋转、亮度/对比度扰动进行数据增强。
- 使用交叉熵损失，并按类别频次设置 class weights，缓解类别不平衡。

## 4. 评估

- 数据划分：训练集、验证集、测试集分离。
- 指标：accuracy、precision、recall、F1-score、confusion matrix。
- 医学场景重点关注恶性类别的 recall，因为漏检恶性样本风险更高。

## 5. 可解释性

- 使用 Grad-CAM 生成热力图。
- 热力图用于观察模型主要关注图像中的哪些区域。
- 该方法不能替代医生诊断，但可以提升模型展示的可解释性。

## 6. 系统展示

- 前端使用 Streamlit。
- 用户上传乳腺超声图像。
- 页面输出预测类别、三类概率、Grad-CAM 热力图和整体测试指标。
- 不单独开发后端，因为 Streamlit 已能完成页面交互和模型推理。

## 7. 局限性

- 数据集规模有限。
- 模型仅用于课程演示，不具备临床诊断资格。
- 后续可以加入更多医院、多设备、多中心数据进行泛化验证。

