import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random
from HandAngle import HandAngle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Read csv file ###

def Readcsv():
    input_path = "" # input_path
    train_df = pd.DataFrame()
    for i in range(10):
        temp_df = pd.read_csv(
                            input_path + 
                            f"{i+1}/" + 
                            f"Output Images - Sign {i+1}.csv") # input csv file name
        train_df = pd.concat([train_df, temp_df])

    return train_df
train_df = Readcsv()

### Define Dataset ###

class ASLDataset(Dataset):
    def __init__(self, train_df):
        landmark = torch.tensor(train_df.iloc[:, 1:64].values).float()
        self.label = torch.tensor(train_df.iloc[:, 64].values - 1).long() # Match labels to start with 0
        self.angle = torch.zeros((len(self.label), int(21*20/2)))

        for i in tqdm(range(len(self.label))):
            # randomly flip x coordinate(for left handed)
            if random.getrandbits(1):
                landmark[i, 0:63:3] = 1 - landmark[i, 0:63:3]
            
            # get angles from landmarks
            self.angle[i] = HandAngle(landmark[i, :])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.angle[idx], self.label[idx]

### train/validate datset/dataloader ###
batch_size = 64
train_dataset, val_dataset = random_split(ASLDataset(train_df), [4000, 1000])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

### Define model ###
class NumberDetector(nn.Module):
    def __init__(self):
        super(NumberDetector, self).__init__()
        # 3 fully connected layers
        self.fc_layer1 = nn.Sequential(
                nn.Linear(210, 100),
                nn.LeakyReLU(),
                nn.Dropout()
                )
        self.fc_layer2 = nn.Sequential(
                nn.Linear(100, 50),
                nn.LeakyReLU(),
                nn.Dropout()
                )
        self.fc_layer3 = nn.Sequential(
                nn.Linear(50, 10)
                )
        
    def forward(self, x):
        output = x.view(-1, 210)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        output = self.fc_layer3(output)
        return output

model = NumberDetector().to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
model.train()

### Trainig ###

import time
start = time.time()
loss = []
for epoch in range(100) :
    print("{}th epoch starting.".format(epoch+1))
    for i, (landmarks, labels) in enumerate(tqdm(train_loader)):
        landmarks, labels = landmarks.to(device), labels.to(device)

        optimizer.zero_grad()
        train_loss = loss_function(model(landmarks), labels)
        train_loss.backward()

        optimizer.step()

    print ("Epoch [{}] Loss: {:.4f}".format(epoch+1, train_loss.item()))
    loss.append(train_loss)

    model.eval()
    val_loss, correct, total = 0, 0, 0

    ### valdation loss ###

    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for i, (landmarks, labels) in enumerate(tqdm(val_loader)):
            landmarks, labels = landmarks.to(device), labels.to(device)

            output = model(landmarks)
            val_loss += loss_function(output, labels).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            total += labels.size(0)

    print('\n[Validation set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss /total, correct, total,
            100. * correct / total))

end = time.time()
print("Time ellapsed in training is: {}".format(end - start))

# Save trained model weights
torch.save(model.state_dict(), "NumberDetector_state_dict.pt")