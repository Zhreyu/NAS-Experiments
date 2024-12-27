import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
def generate_data(num_samples=1000000, num_features=450):
    # Random features
    X = np.random.randn(num_samples, num_features)
    # Random binary labels
    y = np.random.randint(0, 2, size=(num_samples, 1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define a simple neural network with one variable layer
class Net(nn.Module):
    def __init__(self, num_neurons=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Train the model
def train_model(model, data_loader, epochs=5):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# Main NAS routine
def run_nas():
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    num_neurons_options = [5, 10, 20, 50,100,200,500,5000]  # Different architectures to explore
    best_accuracy = 0
    best_architecture = None

    for neurons in num_neurons_options:
        model = Net(num_neurons=neurons)
        trained_model = train_model(model, loader)
        # Evaluate the model (simplified evaluation using the training set)
        with torch.no_grad():
            predictions = trained_model(X).round()
            correct = (predictions == y).float().sum()
            accuracy = correct / len(y)
            print(f"Architecture with {neurons} neurons, Accuracy: {accuracy.item()}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_architecture = neurons

    print(f"Best architecture has {best_architecture} neurons with accuracy {best_accuracy.item()}")

if __name__ == "__main__":
    run_nas()
