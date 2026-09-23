import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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


# Create model
model = NeuralNetwork()

# Load trained weights
model.load_state_dict(
    torch.load("mnist_model.pth")
)

model.eval()


# Load test data
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# Select one image
index = int(input("Enter test image index (0-9999): "))

image, label = test_dataset[index]


# Make prediction
with torch.no_grad():

    output = model(image.unsqueeze(0))

    prediction = output.argmax(dim=1).item()


print("Index:", index)
print("Actual digit:", label)
print("Predicted digit:", prediction)


# Display image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label} | Predicted: {prediction}")
plt.axis("off")
plt.show()