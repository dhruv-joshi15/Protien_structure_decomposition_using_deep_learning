import numpy as np 
import pandas as pd 

import os

!pip3 install ax-platform

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render
import seaborn as sns
import matplotlib.pyplot as plt

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

print("Using CUDA? " + str(use_cuda))
print(device)

seq_train = pd.read_csv('/kaggle/input/deep-learning-for-msc-202324/seqs_train.csv')
labels_train = pd.read_csv('/kaggle/input/deep-learning-for-msc-202324/labels_train.csv')

label_map = {'H': 0,'E': 1,'C': 2}
model_path = '/kaggle/input/deep-learning-for-msc-202324/model.pth'

class ProteinDataset(Dataset) :

    def __init__(self, root_dir, labels) :

        self.path = root_dir
        self.labels = labels
        self.file_name = os.listdir(self.path)

    def __len__(self) :
        return len(self.file_name)

    def __getitem__(self,idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_name[idx]
        file_path = os.path.join(self.path, file_name)   # Find Pssm files
        file = pd.read_csv(file_path)

        pssm = file[file.columns[2:]]                    # First two columns are index and feature name
        X = torch.from_numpy(pssm.values.astype(np.float32))
        pad_shape = 1733 - X.shape[0]
        X = F.pad(X, (0, 0, 0, pad_shape), 'constant', 0)# Padding as [1733, 20]

        label_name = file_name.replace("_train.csv", "")
        label = self.labels[self.labels['PDB_ID'] == label_name]['SEC_STRUCT'].values[0]
        y = [label_map[char] for char in label]
        y += [-1 for i in range(pad_shape)]              # Padding to 1733
        y = torch.from_numpy(np.array(y))


        return X, y

dataset = ProteinDataset('/kaggle/input/deep-learning-for-msc-202324/train', labels_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Set seed
torch.manual_seed(42)

# Use random_split to split dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Step2: Model Definition
class FCN(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate=0.5):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(128, output_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.conv3(x)
        return x.permute(0, 2, 1)

def train_model(train_loader, val_loader, parameterisation, num_epochs=10):
    model = FCN(input_channels=20, output_channels=3, dropout_rate=parameterisation['dropout_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=parameterisation['lr'])
    best_loss = float('inf')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for sequences, labels in tqdm(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.reshape(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 2)

            # Extend the lists with the labels and predictions
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_preds.extend(predicted.view(-1).cpu().numpy())
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

        avg_loss = total_loss / len(train_loader)
#         print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "model.pth")
            print(f"Model saved at Epoch {epoch+1} with loss {avg_loss}")
        
        model.eval()  # Evaluation mode
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, labels in tqdm(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                outputs = model(sequences)

                loss = criterion(outputs.reshape(-1, 3), labels.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss}")

    return best_loss

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# train_loss = train_model(train_loader, val_loader, num_epochs=1, lr = 0.01, dropout_rate = 0.43)

ax_client = AxClient()

ax_client.create_experiment(
    name="FCN_optimization",  # The name of the experiment.
    parameters=[
        {
            "name": "lr",  # The name of the parameter.
            # The type of the parameter ("range", "choice" or "fixed").
            "type": "range",
            "bounds": [1e-6, 1e-2],  # The bounds for range parameters.
            # "values" The possible values for choice parameters .
            # "value" The fixed value for fixed parameters.
            # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
            "value_type": "float",
            # Optional, whether to use a log scale for range parameters. Defaults to False.
            "log_scale": True,
            # "is_ordered" Optional, a flag for choice parameters.
        },
        {
            "name": "dropout_rate",
            "type": "range",
            "bounds": [0.1, 1.0],
        },
    ],
    objectives={"loss": ObjectiveProperties(minimize=True)},
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for i in range(20):  
    parameters, trial_index = ax_client.get_next_trial()
    metrics = train_model(train_loader, val_loader, parameters, num_epochs=5)
    ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)

best_parameters, metrics = ax_client.get_best_parameters()

train_loss = train_model(train_loader, val_loader, best_parameters, num_epochs=20)

result = {}
map_label = {0: 'H', 1: 'E', 2: 'C'}

class test_dataset(Dataset) :

    def __init__(self, pssm_path) :

        self.path = pssm_path
        self.file_name = os.listdir(self.path)

    def __len__(self) :

        return len(self.file_name)

    def __getitem__(self,idx) :

        file_name = self.file_name[idx]
        file_path = os.path.join(self.path, file_name)   # Find Pssm files
        file = pd.read_csv(file_path)

        pssm = file[file.columns[2:]]                    # First two columns are index and feature name
        X = torch.from_numpy(pssm.values.astype(np.float32))
        seq_len = X.shape[0]
        pad_shape = 1733 - X.shape[0]
        X = F.pad(X, (0, 0, 0, pad_shape), 'constant', 0)# Padding as [1733, 20]

        return X, np.array([seq_len]), file_name



def test_model(model, test_loader, output_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for inputs,  seq_len, name in tqdm(test_loader):
        
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 2)

        id_ = name[0].replace('_test.csv', '')

        for j in range(seq_len[0]):
            result[id_ + '_' + str(j + 1)] = map_label[predicted[0, j].item()]
    
    df = pd.DataFrame(list(result.items()), columns=['ID', 'STRUCTURE'])
    df.to_csv('new_prediction.csv', index=False)

test_dataset = test_dataset('/kaggle/input/deep-learning-for-msc-202324/test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = FCN(input_channels=20, output_channels=3, dropout_rate = 0.43)
model.load_state_dict(torch.load('model.pth'))
output_file_path = "/kaggle/input/deep-learning-for-msc-202324/prediction.csv"
model.eval()

test_model(model, test_loader, output_file_path)
