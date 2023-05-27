from Dataset import AirDataSet
import torch
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, optimizer, loss_module, learning_rate=0.001, num_epochs=500, model_name="model"):
  torch.manual_seed(42) # Setting the seed
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
  print("Current device is {}".format(device))
  causes_list = ["508","509","510","511","512","513","514","515","516","520","-1"]
  """
  cause_id columns meaning:
    "508","Chronic respiratory diseases"
    "509","Chronic obstructive pulmonary disease"
    "510","Pneumoconiosis"
    "511","Silicosis"
    "512","Asbestosis"
    "513","Coal workers pneumoconiosis"
    "514","Other pneumoconiosis"
    "515","Asthma"
    "516","Interstitial lung disease and pulmonary sarcoidosis"
    "520","Other chronic respiratory diseases"
    "-1","All causes"
  """
  for cause_id in causes_list:
    print(f"Training for cause_id {cause_id}")

    train_dataset = AirDataSet(cause_id, validation_size=0.2, train=True)
    train_data_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = AirDataSet(cause_id, validation_size=0.2, train=False)
    test_data_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    current_model = model(input_dim=train_dataset.data.shape[1]).to(device)
    current_optimizer = optimizer(current_model.parameters(), lr=learning_rate)

    epochs = []

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
      epochs.append(epoch)

      # Training the model for this one epoch
      current_model.train()
      for data_point, data_label in train_data_loader:
        data_point = data_point.to(device)
        data_label = data_label.to(device)
        data_label = data_label.squeeze(dim=1)

        output = current_model(data_point)
        output = output.squeeze(dim=1)


        loss = loss_module(output, data_label.float())
      
        current_optimizer.zero_grad()
        loss.backward()
        current_optimizer.step()
      train_losses.append(loss.item())
      print(f"Epoch {epoch} Training loss: {loss.item()}")

      # Testing the model
      current_model.eval()
      with torch.no_grad():
        for data_point, data_label in test_data_loader:
          data_point = data_point.to(device)
          data_label = data_label.to(device)
          data_label = data_label.squeeze(dim=1)

          output = current_model(data_point)
          output = output.squeeze(dim=1)

          loss = loss_module(output, data_label.float())
        print(f"Epoch {epoch} Testing loss: {loss.item()}")
        test_losses.append(loss.item())



    # Plotting the loss
    plt.clf()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Testing Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss for cause_id {cause_id} with {model_name}")
    plt.legend()
    plt.savefig(f'./results/{model_name}/cause_id_{cause_id}.png')

    # Saving the model
    torch.save(current_model.state_dict(), f"./results/{model_name}/cause_id_{cause_id}.tar")

    # Saving the losses
    with open(f"./results/{model_name}/cause_id_{cause_id}_losses.txt", "w") as f:
      f.write("Epochs,Training Loss,Testing Loss\n")
      for epoch, train_loss, test_loss in zip(epochs, train_losses, test_losses):
        f.write(f"{epoch},{train_loss},{test_loss}\n")




