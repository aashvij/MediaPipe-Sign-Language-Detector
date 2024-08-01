import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import pandas as pd
import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# this class inhereits Dataset class
class LanguageDataset(Dataset):
    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 0]
        data = self.img_labels.iloc[idx, 1:43]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(np.array(ord(label) - ord('a')), dtype=torch.long)
    
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(42, 256)  # input layer (63) -> hidden layer (128)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)  # input layer (63) -> hidden layer (128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)  # hidden layer (128) -> hidden layer (64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 26)  # hidden layer (64) -> output layer (26)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(func.relu(self.bn1(self.fc1(x))))
        x = self.dropout(func.relu(self.bn2(self.fc2(x))))
        x = self.dropout(func.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x
    
#implementation for GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[-1])
        return out

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            # print("predictions:", pred)
            # print("labels:", y)
            print(f"loss: {loss} [{current}/{size}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
    

#hyperparameters
learning_rate = 0.001
batch_size = 50
epochs = 50

filePath = '/Users/aashvijain/projects/typescriptFirst/hand_landmarks.csv'
testPath = '/Users/aashvijain/projects/typescriptFirst/testingData.csv'
trainingDataset = LanguageDataset(annotations_file=filePath)
testingDataset = LanguageDataset(annotations_file=testPath)
trainDataloader = DataLoader(trainingDataset, batch_size=batch_size, shuffle=True)
testDataloader = DataLoader(testingDataset, batch_size=batch_size, shuffle=True)

model = FCNN().to(device)
#model = GRUModel(input_size = 42, hidden_size = 512, num_layers=5, num_classes=26).to(device)

#Loss Function: L(y, y') = -∑(y * log(y'))
# - sum of observed (1) * log of predicted probability of a certin classification --> comes from KL probability formula
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01) #change learning rate

for epoch in range(epochs):
    print(f"Epoch number {epoch+1}")
    train_loop(trainDataloader, model, loss_fn, optimizer)
    scheduler.step()
    val_loss, correct = test_loop(testDataloader, model, loss_fn)

print("Done")