import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, params['hidden_size'])
        self.fc2 = nn.Linear(params['hidden_size'], 10)
        self.dropout = nn.Dropout(params['dropout_rate'])

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the images
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def get_data_loaders():
    # Transformations and data loading for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def train(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    train_loader = get_data_loaders()
    criterion = nn.NLLLoss()

    # Training loop
    for epoch in range(params['epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Simplified evaluation on training data
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(train_loader.dataset)
    print(f"Accuracy: {accuracy}")
    nni.report_final_result(accuracy)

if __name__ == '__main__':
    try:
        # Get the parameters from NNI
        tuned_params = nni.get_next_parameter()
        train(tuned_params)
    except Exception as exception:
        print(exception)
