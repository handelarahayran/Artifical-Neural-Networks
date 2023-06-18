import math
from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mnist import MNIST
import matplotlib.pyplot as plt
import random


# Simulation Settings (controlled by the user)
n_epochs = 30             # Number of epochs for the training
batch_size = 500            # Batch size for the training
hidden_layers = [120, 350, 50]  # Number of neurons in each hidden layer (both variable)
loss_function = 3           # Loss Function: 1: Mean Squared Error, 2: Cross Entropy , 3: Binary Cross Entropy
learning_rate = 0.00001     # Learning rate for the training


# Import and prepare the data
data = MNIST()
train_data, train_lab = data.load_training()
test_data, test_lab = data.load_testing()
train_data = torch.tensor(train_data, dtype=torch.float32)
train_lab = torch.squeeze(torch.tensor(train_lab, dtype=torch.long).reshape(-1, 1))
test_data = torch.tensor(test_data, dtype=torch.float32)
test_lab = torch.squeeze(torch.tensor(test_lab, dtype=torch.long).reshape(-1, 1))

# Initialize important variables
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)
fig1, ax1 = plt.subplots()
features = 784
test_accu_list = np.zeros(n_epochs)
train_accu_list = np.zeros(n_epochs)
loss_list = np.ones(math.ceil(train_data.size(dim=0)/batch_size))
if loss_function == 1:
    loss_fn = nn.MSELoss()
elif loss_function == 2:
    loss_fn = nn.CrossEntropyLoss()
else:
    loss_fn = nn.BCELoss()

# Prepare the NN model
temporary = features
modules = []
for i in range(len(hidden_layers)):
    modules.append(nn.Linear(temporary, hidden_layers[i]))
    modules.append(nn.ReLU())
    temporary = hidden_layers[i]
modules.append(nn.Linear(temporary, 10))
modules.append(nn.Sigmoid())
model = nn.Sequential(*modules)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Begin training
stopwatch = perf_counter()
for epoch in range(n_epochs):
    pointer = 0
    for i in range(0, len(train_data), batch_size):
        # Get batch data
        data_batch = train_data[i:i + batch_size]
        prediction = model(data_batch)
        lab_batch = train_lab[i:i + batch_size]
        # Compute batch loss
        if loss_function == 2:
            loss = loss_fn(prediction, lab_batch)
        else:
            labels = torch.zeros(data_batch.size(dim=0), 10)
            for j in range(data_batch.size(dim=0)):
                labels[j][int(lab_batch[j])] = 1
            loss = loss_fn(prediction, labels)
        # Apply backward propagation for batch
        loss_list[pointer] = loss
        pointer += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate epoch statistics
    test_pred = model(train_data)
    lab_pred = torch.argmax(test_pred, dim=1)
    train_accu_list[epoch] = 100*sum(lab_pred == train_lab)/len(lab_pred)

    test_pred = model(test_data)
    lab_pred = torch.argmax(test_pred, dim=1)
    test_accu_list[epoch] = 100*sum(lab_pred == test_lab) / len(lab_pred)

    print(f'Finished epoch {epoch + 1}: train set accuracy {train_accu_list[epoch]:.4f},'
          f' test set accuracy {test_accu_list[epoch]:.4f}')

    # Plot Epoch loss
    ax1.cla()
    ax1.plot(loss_list, linewidth=2.0)
    ax1.legend(['epoch ' + str(epoch + 1)])
    ax1.set_title("Loss Function per batch")
    plt.draw()
    plt.pause(0.1)


# Plot final results
timer = perf_counter() - stopwatch
print("Total training time (seconds): " + str(timer))
fig2, (ax2, ax3) = plt.subplots(2)
ax2.set_title("Training set accuracy per epoch")
ax2.plot(train_accu_list,'*-', linewidth=2.0,)
ax3.set_title("Test set accuracy per epoch")
ax3.plot(test_accu_list,'*-', linewidth=2.0)
plt.show()