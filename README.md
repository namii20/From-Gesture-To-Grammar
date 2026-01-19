**From Gesture to Grammar: A Deep Neural Approach to Sign Language Understanding**

**📌 Overview**

A computer vision & deep learning system for ASL recognition, translating static hand gestures (A–Y, excluding J & Z) into text. Supports real-time recognition via webcam to bridge communication between hearing-impaired and non-signers.

**🧠 Key Contributions**

CNN-based model for static ASL gestures

Image preprocessing & augmentation applied

Achieved ~94% accuracy

Real-time webcam recognition

Extensions planned for dynamic gestures & full sign-to-text translation

**🗂️ Dataset**

ASL Alphabet Dataset | 29 classes (A–Z excluding J & Z, plus space, delete, nothing)

Preprocessing: Resize, normalize, background reduction, augmentation

Split: 70% train | 20% val | 10% test

Dataset Link: https://www.kaggle.com/datasets/avnijaiswal/asl-alphabet-dataset

**🏗️ Model**

CNN layers for spatial features

Optional LSTM for temporal patterns (future)

Input: 64×64×3 | Optimizer: Adam (lr=0.001) | Batch: 32 | Epochs: 30–50

Libraries: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib

**📊 Results**

Accuracy: ~94%

Most classes: strong precision, recall, F1

Minor confusion in visually similar gestures (A vs V)

Real-time testing: webcam recognition works well

**⚠️ Limitations**

Dynamic gestures (J & Z) not included

Limited testing across diverse users/environments

Performance may drop in poor lighting/clutter

**🚀 Future Work**

Add dynamic gesture recognition (LSTM/Transformer)

Expand dataset with varied users & conditions

Transfer learning (ResNet, MobileNet, VGG)

Deploy as mobile/web app

Integrate sign-to-speech translation

👩‍💻 Authors

Avani Jaiswal – B.Tech in AI & ML, Indira Gandhi Delhi Technical University for Women (IGDTUW)

Namita Belwal – B.Tech in ECE-AI, Indira Gandhi Delhi Technical University for Women (IGDTUW)
