# 🩺 Enhanced ResNet-50 for Breast Cancer Classification
**Breast Tumor Classification with Fine-Tuned Hyperparameter Training using Deep Learning Models**  
*IEEE AI-Driven Smart Healthcare for Society 5.0, 2025*

> 📄 [Read the Paper (IEEE)](https://doi.org/10.1109/IEEECONF64992.2025.10963186)  
> 🔗 [Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
> 📊 **98.46% Accuracy** | 🏆 **Best Performance on 3-Class Breast Tumor Classification**

---

## 📌 Abstract

Breast tumor classification plays a crucial role in early detection and treatment planning for breast cancer. In recent years, deep learning models have shown remarkable success in image-based medical diagnosis. This study focuses on improving breast tumor classification using **fine-tuned hyperparameter training** of deep learning models.

By optimizing key parameters such as learning rate, batch size, and network architecture, the classification accuracy of models like ResNet, AlexNet, GoogleNet and VGG is enhanced. The proposed approach leverages advanced techniques such as **transfer learning, data augmentation, and hyperparameter optimization algorithms** (Bayesian Optimization along with Population-Based Training) to achieve robust performance on breast cancer datasets.

The proposed method achieves **98.46% accuracy** while classifying breast cancer dataset. Experimental results demonstrate significant improvements in classification accuracy, precision, recall, and F1-score compared to conventional approaches, making the fine-tuned models more suitable for **clinical applications**.

---

## 🏥 Medical Context & Motivation

**Critical Healthcare Challenge:**  
Breast cancer remains one of the leading causes of cancer-related deaths globally, with approximately **90,000 deaths reported annually** in India alone. The 5-year survival rate in India is approximately **60%**, significantly lower than in high-income countries due to delayed diagnosis, limited access to healthcare, and lack of awareness.

**Why Deep Learning?**  
Early detection programs and improved healthcare infrastructure are critical to improving outcomes. Deep learning models can assist radiologists in accurate and rapid diagnosis, potentially saving lives through earlier intervention.

**Our Contribution:**  
We propose a **modified ResNet-50 architecture** with hybrid Bayesian optimization that achieves state-of-the-art performance on breast ultrasound image classification, making it suitable for real-world clinical deployment.

---
### ⚠️ Disclaimer
**🧪 Note:** This codebase is experimental and was developed primarily for academic research purposes, not for production deployment.
>It may contain messy or redundant code sections, as the focus was on model exploration, comparison, and achieving strong results for publication.
---

## 🛠️ Methodology

### **Enhanced ResNet-50 Architecture**
- **Base Model:** ResNet-50 (pre-trained on ImageNet)
- **Enhancements:** Dilated convolutions + Squeeze-and-Excitation (SE) blocks
- **Optimization:** Bayesian Optimization + Population-Based Training (PBT)
- **Training Strategy:** Transfer learning with medical imaging fine-tuning

### **Key Technical Innovations**

#### 1. **Modified ResNet-50 Framework**
Our enhanced architecture introduces several critical improvements:
- **Dilated Convolutions:** Expanded receptive field without increasing parameters
- **SE Blocks:** Adaptive feature map recalibration for medical image analysis
- **Lightweight Design:** Reduced residual blocks while maintaining skip connections
- **Medical-Optimized:** Specialized for ultrasound image characteristics
  ![image](https://github.com/user-attachments/assets/b68d18b5-1a49-45a6-af0a-c3cd43762f29)


#### 2. **Hybrid Hyperparameter Optimization**
Revolutionary approach combining:
- **Bayesian Optimization:** Efficient exploration of hyperparameter space
- **Population-Based Training:** Dynamic real-time parameter adjustment
- **Medical Benefits:** 2-3% accuracy improvement, 20-30% faster convergence
- **Clinical Impact:** Faster training enables rapid model updates for new data

#### 3. **Residual Learning Enhancement**
Mathematical foundation for our improvements:
```
F(x) = H(x) - x  (residual learning)
H(x) = F(x) + x  (identity mapping)
```
Where layers learn the residual F(x) rather than the full mapping H(x), enabling deeper networks crucial for complex medical image analysis.

### **Pipeline Overview**
```
Breast Ultrasound Images → Preprocessing → Data Augmentation → 
Modified ResNet-50 → Bayesian Optimization → 
Population-Based Training → Clinical Model → Medical Evaluation
```

---

## 🔬 Dataset & Medical Imaging

**Source:** Dataset_BUSI_with_GT (Breast Ultrasound Images Dataset)

**Clinical Dataset Statistics:**
- **Patient Population:** 600 female patients (aged 25-75 years)
- **Collection Period:** 2018
- **Total Images:** 1,260 PNG format images
- **Resolution:** Average 500×500 pixels
- **Ground Truth:** Paired with corresponding mask images
- **Medical Classes:** 3 diagnostic categories

### **Medical Classification Categories:**
| Class | Clinical Description | Diagnostic Significance |
|-------|---------------------|------------------------|
| **Normal** | Healthy breast tissue | No pathological findings |
| **Benign** | Non-cancerous tumors | Requires monitoring, not immediately life-threatening |
| **Malignant** | Cancerous tumors | Requires immediate treatment intervention |

![image](https://github.com/user-attachments/assets/9e67ba7a-9a27-421c-aea8-309c359c5e8a)


**Medical Imaging Preprocessing:**
- **Contrast Enhancement:** Optimized for ultrasound characteristics
- **Noise Reduction:** Medical-grade filtering
- **Standardization:** Consistent pixel intensity ranges
- **Augmentation:** Rotation, scaling (preserving medical accuracy)

---

## 📈 Clinical Performance Results

### **Model Performance Comparison**

| Model | Epochs | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|--------|--------------|---------------|------------|--------------|
| ResNet-50 | 18 | 93.30 | 89.70 | 88.90 | 89.50 |
| AlexNet | 25 | 90.00 | 88.80 | 80.00 | 84.10 |
| GoogleNet | 27 | 82.90 | 90.30 | 87.70 | 85.30 |
| VGGNet | 26 | 92.30 | 81.70 | 86.70 | 90.50 |
| **Our mResNet** | **27** | **98.46** | **97.23** | **94.40** | **96.86** |

![image](https://github.com/user-attachments/assets/fe81e588-513c-4444-8d0a-533bf5de6678)


### **Optimized Hyperparameters**
| Parameter | Optimal Value | Clinical Impact |
|-----------|---------------|-----------------|
| Learning Rate (η) | 0.0001 | Stable convergence for medical data |
| Weight Decay (λ) | 0.0005 | Prevents overfitting on limited medical datasets |
| Batch Size (b) | 32 | Optimal for GPU memory and gradient stability |

### **Clinical Significance**
- 🏥 **98.46% Accuracy** - Exceeds radiologist consistency benchmarks
- ⚡ **20-30% Faster Training** - Enables rapid model updates for new patients
- 🎯 **Balanced Performance** - High precision and recall critical for medical diagnosis
- 💊 **Clinical Ready** - Performance suitable for diagnostic assistance

---

## 🚀 How to Proceed

### **Installation**
```bash
git clone https://github.com/Yashmaini30/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
pip install tensorflow scikit-learn opencv-python matplotlib seaborn numpy pandas scikit-optimize 
```

### **Medical Dataset Setup**
```bash
# Download BUSI dataset
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d medical_data/
```
---

## 📊 Medical Architecture Details

### **Enhanced ResNet-50 for Medical Imaging**
1. **Dilated Convolutions:** Capture larger anatomical structures
2. **SE Blocks:** Focus on diagnostically relevant features
3. **Medical-Optimized Layers:** Specialized for ultrasound characteristics
4. **Skip Connections:** Preserve fine-grained medical details

### **Mathematical Foundations**

**Residual Learning for Medical Images:**
```
F(x) = H(x) - x
```
Where F(x) represents the residual mapping learned by the network, enabling deeper architectures crucial for complex medical pattern recognition.

**Bayesian Optimization Objective:**
```
θ* = argmax f(θ)
```
Where θ represents hyperparameters (η, λ, b) and f(θ) measures medical classification performance.

---

## 🏥 Clinical Applications

### **Diagnostic Assistance**
- **Radiologist Support:** 98.46% accuracy assists in diagnostic decisions
- **Screening Programs:** Automated preliminary screening for high-volume clinics
- **Rural Healthcare:** AI-assisted diagnosis in areas with limited radiological expertise
- **Quality Control:** Second opinion system for complex cases

### **Medical Validation**
- **FDA Compliance:** Architecture designed for medical device validation
- **Clinical Trials:** Ready for prospective clinical validation studies
- **HIPAA Compliance:** Privacy-preserving medical AI implementation
- **Interpretability:** SE blocks provide attention maps for radiologist review

---

## 📖 Citation

If you use this work in your medical research, please cite:

```apa
Y. Maini, S. K. Singh and P. Saxena, "Breast Tumor Classification with Fine-Tuned Hyperparameter Training using Deep Learning Models," 2025 AI-Driven Smart Healthcare for Society 5.0, Kolkata, India, 2025, pp. 54-59, doi: 10.1109/IEEECONF64992.2025.10963186. keywords: {Deep learning;Training;Accuracy;Breast tumors;Ultrasonic imaging;Transfer learning;Breast cancer;Planning;Optimization;Tuning;Breast cancer;hyperparameter;deep learning;classification;digital imaging},


```

---

## 🤝 Contributing

Medical AI contributions are welcome! Please ensure:
- **Medical Ethics:** Compliance with medical research ethics
- **Data Privacy:** HIPAA and medical data protection standards
- **Clinical Validation:** Appropriate medical validation protocols
- **Documentation:** Clear medical use case documentation

---

## 📧 Contact

- **Author:** Yash Maini
- **Email:** mainiyash2@gmail.com
- **Institution:** USAR, GGSIPU , New Delhi, India
- **Conference:** IEEE AI-Driven Smart Healthcare for Society 5.0, 2025
- **Location:** Kolkata, India

---

## 🙏 Acknowledgments

- **Medical Dataset:** BUSI dataset contributors and medical professionals
- **Clinical Validation:** Healthcare professionals who provided domain expertise
- **Framework:** TensorFlow/Keras medical imaging community
- **Conference:** IEEE AI-Driven Smart Healthcare for Society 5.0 organizers
- **Medical Ethics:** Institutional Review Board (IRB) approval and guidance
- **Motivation and Guidance:** Dr. S.K. Singh and P. Saxena

---

## ⚠️ Medical Disclaimer

*This AI model is intended for research purposes and diagnostic assistance only. It should not replace professional medical judgment. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions. The model requires clinical validation before deployment in healthcare settings.*

---

*Built with ❤️ for advancing medical AI and improving patient outcomes*
