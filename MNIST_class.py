import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import nn

# Custom dataset class
class MNISTDataset(Dataset):
    def __init__(self, train=True):
        self.data = datasets.MNIST(
            root="data",
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Two fully connected layers, with a ReLU activation
class MultilayerClassifier(nn.Module):
    def __init__(self, hidden_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.fc2(self.relu(self.fc1(x)))
        return out
    
    def save_model(self, file_name): torch.save(self.state_dict(), f"{file_name}.pth")

# Iterate over dataset, backpropagating loss
def train_model(model, train_loader, criterion, optimiser, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    # Constants - change if training is too slow on system
    NUM_EPOCHS = 5
    NUM_CLASSES = 10
    BATCH_SIZE = 100
    HIDDEN_SIZE = 1000
    LEARNING_RATE = 0.001

    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Uses CUDA if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultilayerClassifier(hidden_size=HIDDEN_SIZE).to(device)

    # Cross entropy loss with Adam optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_model(model, train_loader, criterion, optimiser, device)
        print(f'Epoch {epoch} completed.')

    accuracy = test_model(model, test_loader, device)
    print('Accuracy of the neural network on the 10000 test digits: {}%'.format(accuracy))

    model.save_model("MNIST_classifier")

