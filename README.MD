# MNIST Digit Classifier with PyTorch

A simple feedforward neural network built with **PyTorch** to classify handwritten digits from 0 to 9 using the MNIST dataset.

The model achieves **97.46% test accuracy** on the MNIST test dataset.

## Project Overview

This project demonstrates the basic workflow of building and training a neural network with PyTorch:

```text
MNIST Dataset
     ↓
DataLoader
     ↓
Neural Network
     ↓
Training
     ↓
Evaluation
     ↓
Prediction
```

The model takes a 28×28 grayscale handwritten digit image and predicts which digit it represents.

## Model Architecture

The neural network is a simple fully connected feedforward network:

```text
Input Image
28 × 28 pixels
      ↓
Flatten
784 values
      ↓
Linear Layer
784 → 128
      ↓
ReLU
      ↓
Linear Layer
128 → 10
      ↓
Predicted Digit
0–9
```

### Architecture Details

| Component | Details |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten | 784 input features |
| Hidden Layer | 128 neurons |
| Activation | ReLU |
| Output Layer | 10 neurons |
| Loss Function | Cross Entropy Loss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 5 |
| Batch Size | 64 |
| Test Accuracy | **97.46%** |

## Dataset

The project uses the **MNIST handwritten digit dataset**.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Classes: 10
- Classes represent digits: 0–9

## Project Structure

```text
mnist-pytorch-classifier/
│
├── train.py              # Train and evaluate the model
├── predict.py            # Make predictions on test images
├── mnist_model.pth       # Trained PyTorch model
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Files excluded from Git
│
├── data/                 # MNIST dataset (not tracked)
└── venv/                 # Virtual environment (not tracked)
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Anavyarai/mnist-pytorch-classifier.git
cd mnist-pytorch-classifier
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run:

```bash
python train.py
```

The script downloads the MNIST dataset, trains the neural network, evaluates it on the test set, and saves the trained model as:

```text
mnist_model.pth
```

Example output:

```text
Epoch 1/5, Loss: ...
Epoch 2/5, Loss: ...
Epoch 3/5, Loss: ...
Epoch 4/5, Loss: ...
Epoch 5/5, Loss: ...

Test Accuracy: 97.46%
Model saved successfully!
```

## Making Predictions

Run:

```bash
python predict.py
```

The program asks you to choose an image from the MNIST test dataset:

```text
Enter test image index (0-9999): 250
```

It then displays the selected image and prints:

```text
Index: 250
Actual digit: 6
Predicted digit: 6
```

## Key PyTorch Concepts Demonstrated

This project covers the fundamental PyTorch workflow:

- Tensors
- Datasets
- DataLoaders
- `nn.Module`
- Linear layers
- ReLU activation
- Forward pass
- Cross-entropy loss
- Adam optimizer
- Backpropagation
- Training loops
- Model evaluation
- Saving and loading model weights

## Future Improvements

Possible extensions include:

- Build a graphical interface for drawing digits
- Add prediction confidence
- Visualize incorrect predictions
- Experiment with different network architectures
- Replace the feedforward network with a CNN
- Compare model performance across different hyperparameters

## Technologies

- Python
- PyTorch
- Torchvision
- Matplotlib

## Result

The trained neural network achieved:

**97.46% test accuracy**

on the MNIST test dataset.