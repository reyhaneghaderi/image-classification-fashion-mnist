
#  Fashion MNIST Classification with TensorFlow

This project is a deep learning-based image classifier built with TensorFlow and Keras. It classifies grayscale images of clothing into 10 categories, using the **Fashion MNIST** dataset.

##  Project Objectives

- Learn to load and preprocess image datasets with Keras.
- Build and train a neural network for image classification.
- Visualize model performance and predictions.
- Understand the basics of evaluation metrics like accuracy.

##  Technologies Used

- Languages: Python 3
- Frameworks: TensorFlow, Keras, NumPy, Matplotlib

##  Dataset

- **Fashion MNIST**: 60,000 training and 10,000 test images of 28x28 grayscale clothing items.
- 10 categories: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.

##  How It Works

1. **Data Loading**:
   ```python
   from tensorflow.keras.datasets import fashion_mnist
   (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
   ```
2. **Data Preprocessing**:
   - Normalize pixel values from `[0, 255]` to `[0.0, 1.0]`
   - Convert labels to one-hot encoded format

3. **Model Architecture**:
   - Flatten layer (28x28 → 784)
   - Dense (128) + ReLU
   - Dropout for regularization
   - Output layer (10 classes) + Softmax

4. **Training**:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy

5. **Evaluation & Visualization**:
   - Training vs Validation accuracy/loss plots
   - Predictions on test samples with true vs predicted labels

##  Results

Achieved strong performance using a simple dense neural network:
- **Training Accuracy**: ~88.4%
- **Test Accuracy**: ~86.9%
- **Validation Accuracy**: ~87.6%
 # Key Skills Demonstrated
   -Data preprocessing (normalization, one-hot encoding)
   -Neural network design & training with TensorFlow/Keras
   -Regularization (Dropout)
   -Model evaluation (accuracy plots, confusion matrix)
   -Visualization of predictions
   -Reproducibility with Jupyter Notebooks
   
 #  Research & Future Work
    -Extend to CNNs / ResNets for higher accuracy.
    -Apply interpretability tools (Grad-CAM, SHAP) to analyze learned features.
    -Explore fairness and bias detection across clothing categories.
     -Use project as a teaching baseline for explainable computer vision.
 #    How to Run
    git clone https://github.com/reyhaneghaderi/fashion-mnist-classification
    cd fashion-mnist-classification
    pip install -r requirements.txt
    jupyter notebook fashin_minist.ipynb
 # About Me
👩‍🎓 Master’s student in Stochastic & Data Science, University of Turin
🔍 Research focus: Computer Vision, Contrastive Learning, Federated Learning   



## What I Learned

- Hands-on experience with TensorFlow and neural networks
- Best practices in image data preprocessing
- How to evaluate and debug classification models
- How to visualize and interpret results

 # More Projects: Customer Clustering
, Bank Marketing Prediction
, Street-Style Segmentation


