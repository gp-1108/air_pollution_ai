from Model_Trainer import train_model
import torch.nn as nn
import torch

class LinearClassifier(nn. Module):
  def __init__(self, input_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    x = x.to(self.linear.weight.dtype)
    x = self.linear(x)
    return x

train_model(
  model=LinearClassifier,
  optimizer=torch.optim.SGD,
  loss_module=nn.MSELoss(),
  learning_rate=0.0001,
  num_epochs=100,
  model_name="Linear_Classifier" # It must correspond to the path ./models/{model_name} already existing
)