# 🌄 Scene Classification — GPU-Optimized MobileNetV2 Transfer Learning  
This project builds a full deep learning pipeline to classify **natural scene images** (`buildings, forest, glacier, mountain, sea, street`) using **MobileNetV2** transfer learning in TensorFlow/Keras. It focuses on clean structure, strong generalization, and GPU-accelerated fine-tuning for reliable real-world performance.

🚀 **Key Features**  
**Data Handling & Structure**  
- Automatically loads and preprocesses images using `ImageDataGenerator` with `preprocess_input`.  
- Splits data into 80% training and 20% validation with consistent class ordering.  
- Supports dataset structure:  
  `seg_train/seg_train/{buildings, forest, glacier, mountain, sea, street}`  
  `seg_test/seg_test/{buildings, forest, glacier, mountain, sea, street}`  

**Training Pipeline**  
- **Phase 1:** Trains classifier head (base frozen) using Adam(1e-3) and EarlyStopping.  
- **Phase 2:** Fine-tunes top 60 layers of MobileNetV2 with Adam(1e-6), ReduceLROnPlateau, and ModelCheckpoint.  
- Includes real-time augmentation (rotation, shift, zoom, shear, flip) to improve robustness and prevent overfitting.  

**Model Parameters**  

| Parameter | Value | Description |
|------------|--------|-------------|
| Input Size | 224×224 | Image resolution |
| Batch Size | 32 | Training batch size |
| LR (Head / FT) | 1e-3 / 1e-6 | Learning rates for both phases |
| Dropout | 0.3 | Regularization to avoid overfitting |
| Backbone | MobileNetV2 | Pretrained on ImageNet |
| Optimizer | Adam | Adaptive learning optimization |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint | Training stability |

📊 **Results Summary**  
- **Validation Accuracy:** 88.09%  
- **Test Accuracy:** 87.30%  
- **Validation Loss:** 0.3303  
- **Test Loss:** 0.3349  
- **Correct Predictions:** 2619 / 3000  
All predicted test images are automatically saved to `Predicted_Images/`, organized by their predicted class.

🧠 **Tech Stack**  
Language: Python 3.10+  
Framework: TensorFlow / Keras  
Libraries: NumPy, Pillow, Matplotlib  
Hardware: NVIDIA GPU (CUDA-enabled)  
Model: MobileNetV2 (ImageNet pretrained)

💾 **Output**  
Predicted images saved under:  
`C:/Users/Asus/Downloads/Tensor/New folder/Predicted_Images/`  
Each class (e.g., `buildings/`, `forest/`, `sea/`) contains its predicted samples.  
Trained model checkpoint saved as `best_mnv2.keras`.  

📚 **Author**  
Sina Firoozian 
📧 [sina.firuzian@gmail.com]
