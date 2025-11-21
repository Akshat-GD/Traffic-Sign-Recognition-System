# Traffic Sign Prediction System

## 1. Project Overview

Accurate traffic sign recognition is critical for safe autonomous vehicle operation. Real-world studies demonstrate that robust traffic sign detection systems directly improve autonomous driving safety by ensuring vehicles reliably obey road regulations. CNN-based sign recognition has proven particularly valuable in reducing accidents caused by missed or misinterpreted signage—a frequent failure mode in early autonomous systems.<br> 
This project implements a CNN-based traffic sign classifier. The network processes labeled traffic sign images and outputs a predicted class for each input.<br>
To improve the model’s performance and reliability, **Dropout Regularization** and the **EarlyStopping** callback were incorporated. Together, these techniques help build a robust and generalizable traffic sign classifier that learns meaningful patterns without overfitting. CNN-based TSR models typically achieve near human-level accuracy on benchmark datasets, making them dependable components in autonomous vehicle perception systems and contributing to safer on-road decision-making.

## 2. Dataset Information
This data set was taken from Kaggle, The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.<br>
The dataset has the following properties:
- Single-image, multi-class classification problem
- More than 40 classes
- More than 50,000 images in total
- Large, lifelike database

## 3. Methodology
This project employs a structured deep-learning pipeline that includes the chosen tech stack, a comprehensive EDA to understand dataset characteristics, systematic data preprocessing, a well-designed CNN architecture, a clearly defined training procedure, and regorous evaluation metrics to assess model preformance and generalization.<br>

**3.1. Tech stack employed:**
---
*Core Libraries*<br>
| Library                   | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| Keras                     | Deep learning framework for model building and training |
| Numpy                     | Numerical computing and array operation                 |
| Pandas                    | Data loading, exploration, and manipulation             |
| Scikit-Learn              | Machine learning utilities and preprocessing            |
| mlextend                  | Extended machine learning utilities (evaluation)        |
| OpenCV (cv2)              | Image processing and resizing                           |
| PIL (Pillow)              | Image file handling and manipulation                    |

*Visualization Libraries*<br>

| Library                   | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| Matplotlib                | Plotting training curves and visualization              |
| Seaborn                   | Statistical visualization (KDE plots, histogram)        |
    
*System Libraries*<br>

| Libraries                 | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| os                        | File system operations and directory management         |
| shutil                    | File and directory                                      |

*Development Enviornmet*<br>

- Google colab

**3.2. Exploratory Data Analysis Report**<br>
---
The GTSRB dataset presents a diverse and challenging real-world benchmark, closely reflecting practical conditions encountered by traffic sign recognition systems. key insights from the EDA include:<br>
- *Class Distribution*:<br>
Histogram analyses shows significant variation in the number of samples across different traffic sign categories, highlighting an imbalanced class structure.<br>

- *Image Resolution Variability*:<br>
Original images span a wide resolution, from approximately 35x35 px to 100x100 px, introducing natural diversity and contributing to the dataset's realism.<br>

**3.3. Data Preprocessing**<br>
---
We take each image (from their respective folders), resize it to 30x30 px (as this was found to be the most observed resolution), convert it into array and append it to a single list which is then later converted to an array. This is then split into training and testing sets.

Raw Images(variable resolution) -> Resize to 30x30 px -> Convert to Arrays -> Train-Test Split(80:20) -> Feed to Model

**3.4. Model Architecture**<br>
---
The network follows a sequential CNN design with stacked convolutional blocks, pooling layers and fully connected layers for final classification. The architecture is structured as follows:<br>

|       Layer       |          Type         |         Configuration         |
|-------------------|-----------------------|-------------------------------|
| 1                 | Conv2D                | 32 filters, 5x5 kernel, ReLU  |
| 2                 | Conv2D                | 32 filters, 5x5 kernel, ReLU  |
| 3                 | MaxPool2D             | 2x2 pool size                 |
| 4                 | Dropout               | rate = 0.25                   |
| 5                 | Conv2D                | 64 filters, 3x3 kernel, ReLU  |
| 6                 | Conv2D                | 64 filters, 3x3 kernel, ReLU  |
| 7                 | MaxPool2D             | 2x2 pool size                 |
| 8                 | Dropout               | rate = 0.25                   | 
|Flattening layer   | -                     | -                             |
| 9                 | Dense                 | 256 units, ReLU               |
| 10                | Dropout               | rate = 0.5                    |
| 11                | Dense                 | 43 units, softmax             |

**Design choices explained**:<br>
*Convolutional Layers*:<br>
Captures hierarchical spatial features such as edges, shapes and color patterns essential for identifying traffic signs

*Max Pooling*:<br>
reduces spatial dimensions while retaining the most informative features, improving both efficiency and generalization.

*Dropout Regularization*:<br>
Mitigates overfitting by randomly deactivating neurons during training.
- 0.25 rate in early layers: Preserves low-level features richness while still providing regularization.
- 0.5 rate in dense layers: Offers stronger regularization where overfitting is more likely.

*Softmax Activation*:<br>
Produces a probability distribution over all 43 classes for final classification.

*Sequential Architecture*:<br>
Allows progressive feature extraction, from basic edges to complex sign patterns mirroring aspects of human visual processing.

**3.5. Training Procedure**
---
The model was trained using the following configurations:<br>

| Parameter         | Configuration                      |
|-------------------|------------------------------------|
| Batch size        | 32                                 | 
| Epochs            | 30                                 | 
| Optimizer         | Adam                               | 
| Loss function     | Categorical Crossentropy           |
| Early Stopping    | (patience = 8, min_delta = 0.02)   |

*Training insights*

- Training automatically halts at epoch 13.

**3.6. Model Evaluation Metrics**
---

The model's performance is assessed using key evaluation metrics and visual analyses to ensure reliability and generalization.

- *Accuracy Score*:<br>
    Measures the overall correctness of predictions across all classes.

- *Training Vs Validation Accuracy plot*:<br>
    Visualizes learning behavior across epochs, helping identify convergence patterns, underfitting, or overfitting trends.

- *Confusion Matrix*:<br>
    Provides a detailed breakdown of correct and incorrect predictions across all 43 classes, highlighting class-wise performance and potential misclassification patterns.

## **4. Results**    
The model demonstrates strong performance on both training and testing datasets:

- Training Accuracy: 94.82%
- Testing Accuracy: 96.00%

These results indicate that the network generalizes well to unseen traffic sign images, with only a modest performance gap between training and testing accuracy.

## **5. Folder Structure**

```
Traffic_Sign_Recognition/
|
|── traffic_sign_recognition.ipynb # Main google colab notebook
|
|── README.md                      # Project overview and documentation
|
|── requirements.txt               # Python dependencies
```

## **6. Future Improvements**

*Data Augmentation*:<br>
Implement rotation, brightmess adjustment, and perspective tranforms to improve robustness.

*Normalization*:<br>
Apply pixel normalization (0-1 or -1 to 1 range) for improved training stability.

*Model Optimization*:
Experiment with modern achitectures (ResNet, MobileNet) for comparison.

*Real-time Deployment*:
Convert to ONNX or TensorFlow Lite for edge device deployment

*Class Imbalance Handling*:
Apply weighted loss function or SMOTE for underrepresented classes

*Explainability*:
Implement Grad-CAM visualization to understand model predictions.

*Extended Dataset*:
Train on additional traffic sign datasets from different regions.

## **Author**<br>
I am an aspiring Machine Learning Engineer currently pursuing my B.Tech in Artificial Intelligence and Machine Learning, and I’m in my third year of study. I’m passionate about building practical AI solutions, continuously improving my skills, and exploring real-world applications of machine learning. I’m actively looking to collaborate on meaningful projects and open to internship opportunities where I can contribute, learn, and grow alongside experienced professionals.<br>
- GitHub: https://github.com/Akshat-GD
- LinkedIn: https://www.linkedin.com/in/akshat-dandur-a3817932a/
- Email: akshatdandur@gmail.com

*Feel free to reach out for collaboration, feedback, or just to talk tech!*

