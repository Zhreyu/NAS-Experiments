import torch
from torchvision import transforms
from PIL import Image
import sys

import torch.nn as nn

# Define the neural network architecture (same as the one used during training)
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

def load_model(model_path, params):
    model = Net(params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
    return prediction.item()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Define the parameters used during training
    params = {
        'hidden_size': 128,  # Example value, should be the same as used during training
        'dropout_rate': 0.5  # Example value, should be the same as used during training
    }

    model = load_model(model_path, params)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor)
    print(f"Predicted class: {prediction}")