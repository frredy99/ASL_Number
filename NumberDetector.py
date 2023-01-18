import torch.nn as nn

# SignDetector model
class NumberDetector(nn.Module):
    def __init__(self):
        super(NumberDetector, self).__init__()
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
                nn.Linear(50, 10),
                nn.Softmax(dim=1)
                )
        
    def forward(self, x):
        output = x.view(-1, 210)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        output = self.fc_layer3(output)
        return output

if __name__ == "__main__":
    print("NumberDetector module is running.")