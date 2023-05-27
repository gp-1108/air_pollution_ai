from Model_Trainer import train_model
import torch.nn as nn
import torch

class PolynomialClassifier(nn.Module):
  def __init__(self, degree):
    super().__init__()
    self.degree = degree
    self.poly = None

  def set_input_dim(self, input_dim):
    self.poly = nn.Linear(input_dim*(self.degree+1), 1)
    return self
  
  def forward(self, x):
    x_poly = torch.cat([x**i for i in range(self.degree+1)], dim=1)
    return self.poly(x_poly)


for degree in range(2, 10):
  print(f"Training Polynomial Classifier of degree {degree}")
  PartialPolynomialClassifier = PolynomialClassifier(degree=degree)
  learning_rate = pow(10, -(degree+3))
  train_model(
    model=PartialPolynomialClassifier.set_input_dim,
    optimizer=torch.optim.SGD,
    loss_module=nn.MSELoss(),
    learning_rate=learning_rate,
    num_epochs=150,
    model_name=f"Polynomial_Classifier/degree_{degree}" 
  )
