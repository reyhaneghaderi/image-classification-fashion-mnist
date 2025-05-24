
# 👗 Fashion MNIST Classification with TensorFlow

This project is a deep learning-based image classifier built with TensorFlow and Keras. It classifies grayscale images of clothing into 10 categories, using the **Fashion MNIST** dataset.

## 📌 Project Objectives

- Learn to load and preprocess image datasets with Keras.
- Build and train a neural network for image classification.
- Visualize model performance and predictions.
- Understand the basics of evaluation metrics like accuracy.

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib

## 📂 Dataset

- **Fashion MNIST**: 60,000 training and 10,000 test images of 28x28 grayscale clothing items.
- 10 categories: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.

## 🚀 How It Works

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

## 📈 Results

Achieved strong performance using a simple dense neural network:
- **Training Accuracy**: ~87.1%
- **Test Accuracy**: ~87.5%
- **Validation Accuracy**: ~88.3%


## 📚 What I Learned

- Hands-on experience with TensorFlow and neural networks
- Best practices in image data preprocessing
- How to evaluate and debug classification models
- How to visualize and interpret results


