# AI-Based Marine Debris Detection and Classification Using Computer Vision 🌊🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s--seg-red)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)

## 🌊 Project Overview
This project addresses the global environmental crisis of marine plastic pollution by providing a high-speed, high-accuracy **Instance Segmentation** pipeline designed for **Autonomous Underwater Vehicles (AUVs)**. Unlike standard object detection, this system identifies the exact pixel boundaries of debris, enabling precise robotic grasping in challenging underwater environments.

### Key Highlights:
* **Architecture:** YOLOv8s-seg (Small Instance Segmentation model).
* **Accuracy:** 94.8% Detection mAP | 94.0% Mask mAP.
* **Inference Speed:** 12.4ms (Real-time capable).
* **Environment:** Optimized for low-visibility, turbid underwater conditions.

---

## 🏗️ Architectural Pipeline
The "Champion Model" follows a multi-stage process to ensure robustness against underwater noise:

1.  **Input:** Raw RGB Underwater Image (suffering from color cast and turbidity).
2.  **Preprocessing:** **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is used to clear turbidity and sharpen morphological edges.
3.  **Backbone & Neck:** YOLOv8s (CSPDarknet + PANet) for multi-scale feature extraction.
4.  **Dual-Head Processing:** * **Branch A (Detection Head):** Bounding box and class prediction.
    * **Branch B (ProtoNet Head):** Generation of 32 prototype masks (stencils).
5.  **Post-Processing:** **Non-Maximum Suppression (NMS)** to remove redundant overlapping masks.
6.  **Output:** Pixel-perfect segmentation result ready for robotic interaction.

---

## 🧪 Experiments & Ablation Study
We conducted extensive testing to find the optimal configuration:
* **Preprocessing:** Evaluated Masking, Inpainting, and Canny Edge Detection; **CLAHE** emerged as the champion path.
* **Model Scaling:** Compared YOLOv8-Nano vs. **YOLOv8-Small**. Small provided the depth required for complex underwater textures.
* **Comparative Analysis:** Outperformed larger models like SAM (Segment Anything) and RT-DETR in underwater noise handling.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV
* **Deep Learning:** PyTorch, Ultralytics (YOLOv8)
* **Deployment:** Streamlit (Web Dashboard)
* **Training:** Kaggle (Dual NVIDIA Tesla T4 GPUs)

---

## 📊 Results
The model effectively eliminates false positives caused by "Biological Mimicry" through a hybrid training strategy (Debris + Marine Life datasets).

| Metric | Score |
| :--- | :--- |
| **Detection mAP@50** | 94.8% |
| **Mask mAP@50** | 94.0% |
| **Inference Latency** | 12.4 ms |

---

## Component Breakdown
**Preprocessing (CLAHE):** Contrast Limited Adaptive Histogram Equalization neutralizes turbidity and sharpens morphological edges.

**Backbone & Neck (YOLOv8s):** Modified CSPDarknet and PANet extract deep semantic features to distinguish synthetic trash from organic marine life.

**Dual-Head Processing:** Parallel processing of bounding box/class predictions and prototype mask generation.

**Post-Processing (NMS):** Non-Maximum Suppression cleans up duplicate overlapping detections.

## 🚀 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/marine-debris-detection.git](https://github.com/your-username/marine-debris-detection.git)

2. **Create a virtual environment (optional but recommended)**
   
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

4. **Install dependencies**
   
pip install -r requirements.txt

5. **Run the Streamlit Dashboard**
   
streamlit run app.py
