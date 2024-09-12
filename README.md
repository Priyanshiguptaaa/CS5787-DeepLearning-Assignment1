# CS5787-DeepLearning-Assignment1

This project demonstrates how to train and test the LeNet-5 model with various regularization techniques on the FashionMNIST dataset using PyTorch. The regularization techniques evaluated are:

- No Regularization
- Dropout
- Weight Decay
- Batch Normalization

## Prerequisites

Ensure you have the following software installed:

- Python 3.x
- PyTorch
- torchvision
- matplotlib (for plotting)
- pandas (for summarizing results)

## Setup

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Priyanshiguptaaa/CS5787-DeepLearning-Assignment1.git
cd CS5787-DeepLearning-Assignment1
```

### Install Dependencies

```bash
pip install torch torchvision matplotlib pandas
```
### Training Instructions

## Prepare Data
The dataset is assumed to be already preprocessed and loaded into PyTorch DataLoaders (train_loader, val_loader, and test_loader).

## No Regularization
To train the model with no regularization, use the following commands:
```
import torch
import torch.optim as optim
from model import LeNet5  # Replace with the actual model import

# Initialize model
model_no_reg = LeNet5()
model_no_reg.to(device)

# Define optimizer
optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001)

# Train the model
train_loss_no_reg, train_acc_no_reg, val_acc_no_reg, test_acc_no_reg = train_model(
    model_no_reg, optimizer_no_reg, criterion, train_loader, val_loader, test_loader, num_epochs=10, use_dropout=False, save_path='model_no_reg'
)
```

## Dropout
To train the model with Dropout, use the following commands:
```
import torch
import torch.optim as optim
from model import LeNet5Dropout  # Replace with the actual model import

# Initialize model
model_dropout = LeNet5Dropout()
model_dropout.to(device)

# Define optimizer
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)

# Train the model
train_loss_dropout, train_acc_dropout, val_acc_dropout, test_acc_dropout = train_model(
    model_dropout, optimizer_dropout, criterion, train_loader, val_loader, test_loader, num_epochs=10, use_dropout=True, save_path='model_dropout'
)
```
## Weight Decay
To train the model with different weight decay values, use the following commands:

```
import torch
import torch.optim as optim
from model import LeNet5

weight_decays = [1e-5, 1e-4, 1e-3, 1e-2]
for wd in weight_decays:
    # Initialize model
    model_wd = LeNet5()
    model_wd.to(device)

    # Define optimizer
    optimizer_wd = optim.Adam(model_wd.parameters(), lr=0.001, weight_decay=wd)

    # Train the model
    train_loss_wd, train_acc_wd, val_acc_wd, test_acc_wd = train_model(
        model_wd, optimizer_wd, criterion, train_loader, val_loader, test_loader, num_epochs=10, use_dropout=False, save_path=f'model_wd_{wd}'
    )
```

## Batch Normalization
To train the model with Batch Normalization, use the following commands:
```
import torch
import torch.optim as optim
from model import LeNet5BatchNorm  # Replace with the actual model import

# Initialize model
model_bn = LeNet5BatchNorm()
model_bn.to(device)

# Define optimizer
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.001)

# Train the model
train_loss_bn, train_acc_bn, val_acc_bn, test_acc_bn = train_model(
    model_bn, optimizer_bn, criterion, train_loader, val_loader, test_loader, num_epochs=10, use_dropout=False, save_path='model_bn'
)
```

### Testing with Saved Weights
To test a model with saved weights, follow these steps:
```
import torch
from model import LeNet5  # Replace with the actual model import

model_no_reg = LeNet5()
model_no_reg.load_state_dict(torch.load('models/model_no_reg_epoch_X.pth')) # Replace with the model you want to load
model_no_reg.to(device)
model_no_reg.eval()
```

