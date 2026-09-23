import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = NeuralNetwork()

print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
images, labels = next(iter(train_loader))

predictions = model(images)

print("Input shape:", images.shape)
print("Output shape:", predictions.shape)

epochs = 5

for epoch in range(epochs):

    model.train()

    for images, labels in train_loader:

        # 1. Make predictions
        predictions = model(images)

        # 2. Calculate the loss
        loss = loss_function(predictions, labels)

        # 3. Clear previous gradients
        optimizer.zero_grad()

        # 4. Calculate gradients
        loss.backward()

        # 5. Update model weights
        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item():.4f}"
    )

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        predicted_labels = predictions.argmax(dim=1)

        total += labels.size(0)

        correct += (predicted_labels == labels).sum().item()


accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), "mnist_model.pth")

print("Model saved successfully!")
