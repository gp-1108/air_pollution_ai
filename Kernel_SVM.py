from Model_Trainer import train_model
import torch.nn as nn
import torch

class KernelSVM(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self._w = None

    def set_input_dim(self, input_dim):
        num_c = input_dim
        if self.kernel == "linear":
            self._kernel = self.linear
        elif self.kernel == "rbf":
            self._kernel = self.rbf
        self.gamma = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
        self._w = nn.Linear(in_features=num_c, out_features=1)
        return self

    def linear(self, x):
        return x
    
    def rbf(self, x, gamma = 1):
        pairwise_distance = torch.cdist(x, x)
        return torch.exp(-gamma * pairwise_distance ** 2)

    def forward(self, x):
        y = self._kernel(x)
        y = self._w(y)
        return y

# List of kernel functions
kernels = [
    'rbf',
    'linear',
]

for kernel_name in kernels:
    print(f"Training Kernel SVM with {kernel_name} kernel")
    PartialKernelSVM = KernelSVM(kernel=kernel_name)
    train_model(
        model=PartialKernelSVM.set_input_dim,
        optimizer=torch.optim.SGD,
        loss_module=nn.MSELoss(),
        learning_rate=0.01,
        num_epochs=150,
        model_name=f"Kernel_SVM/{kernel_name}"
    )
