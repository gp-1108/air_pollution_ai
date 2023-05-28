from Model_Trainer import train_model
import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def set_input_dim(self, input_dim):
        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        return self
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

for hidden_size in range(1, 10):
    print(f"Training Neural Network with hidden size {hidden_size}")
    PartialNeuralNetwork = NeuralNetwork(hidden_size=hidden_size, output_size=1)
    train_model(
        model=PartialNeuralNetwork.set_input_dim,
        optimizer=torch.optim.SGD,
        loss_module=nn.MSELoss(),
        learning_rate=0.0001,
        num_epochs=150,
        model_name=f"Neural_Network/hidden_size_{hidden_size}" 
    )