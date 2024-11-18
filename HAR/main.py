import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Dataset class
class ConveyorDataset(Dataset):
    def _init_(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                            if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        if self.transform:
            img = self.transform(img)
        return img

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def _init_(self):
        super(Autoencoder, self)._init_()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset and DataLoader
train_data_dir = "path_to_conveyor_images_without_child"
test_data_dir = "path_to_test_images"

train_dataset = ConveyorDataset(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ConveyorDataset(test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, Loss, and Optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training started...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs in train_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "autoencoder.pth")

# Load the model (if needed)
# model.load_state_dict(torch.load("autoencoder.pth"))

# Anomaly Detection
def detect_anomalies(model, test_loader, threshold=0.01):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for idx, img in enumerate(test_loader):
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img).item()
            if loss > threshold:
                anomalies.append((idx, loss))
                # Show the anomalous image
                img_np = img[0].permute(1, 2, 0).cpu().numpy()
                plt.imshow(img_np)
                plt.title(f"Anomaly Detected (Loss: {loss:.4f})")
                plt.show()
    return anomalies

# Detect anomalies
threshold = 0.01  # Adjust this based on validation
anomalies = detect_anomalies(model, test_loader, threshold)
for anomaly in anomalies:
    print(f"Anomaly detected in image {anomaly[0]} with loss {anomaly[1]:.4f}")