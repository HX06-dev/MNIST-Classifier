# MNIST Classifier

A small PyTorch neural network that classifies handwritten digits from the MNIST dataset.

## Setup

```bash
pip install torch torchvision matplotlib
```

## Run

```bash
python train.py
```

The script downloads MNIST into `data/`, trains the model, prints the loss and test accuracy, and saves the learned weights to `model.pt`.

## Configuration

Training settings are defined at the top of `train.py`:

- Batch size: `64`
- Epochs: `5`
- Learning rate: `0.001`
