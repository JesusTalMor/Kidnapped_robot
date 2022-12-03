import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from customDataset import kidnappedDataset

# LeNet Architecture
# 1x32x32 Input -> (5x5), s=1, p=0 -> Avg Pool s=2, p=0 -> (5x5), s=1, p=0 -> Avg pool s=2, p=0
# -> Conv 5x5 to 120 Channels x Linear 120 -> 84 x Linear 10


# * Creating the Neural Net
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.relu = nn.ReLU()
    self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn3 = nn.BatchNorm2d(32)
    self.linear1 = nn.Linear(4608,83)
    self.linear2 = nn.Linear(83,2)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.pool(x)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.linear1(x))
    x = self.linear2(x)
    return x


# * Check if the neural net runs
# x = torch.randn(64, 1, 128, 128)
# model  = LeNet()
# print(model(x).shape)
# exit()

# * Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Checking for Device: {device}')


# * Load Data
print('Loading DataSets')
batch_size = 100
my_transform = transforms.Compose([
  transforms.ToPILImage(), 
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomRotation(degrees=30),
  transforms.ToTensor()
])
datasets = kidnappedDataset(csv_file= 'kidnaped_dataset_modkidnapped.csv', root_dir='images', transform=my_transform)
# print(len(datasets))
train_len = int(len(datasets)*0.9)
test_len = len(datasets) - train_len
print(train_len)
print(test_len)
# print(final_len == len(datasets))
train_set, test_set = torch.utils.data.random_split(datasets, [train_len, test_len])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

def save_checkpoint(state, filename='neural_net.pth.tar'):
  print("=> Saving Checkpoint")
  torch.save(state, filename)

# * Accuracy function
def accuracy(model, loader):
  num_correct = 0
  num_total = 0
  model.eval()
  model = model.to(device=device)
  with torch.no_grad():
    for xi, yi in loader:
      xi = xi.to(device=device, dtype=torch.float32)
      yi = yi.to(device=device, dtype=torch.long)
      scores = model(xi)
      _, pred = scores.max(dim=1)
      num_correct += (pred == yi).sum()
      num_total += pred.size(0)
    return float(num_correct)/num_total


# * Train the network
def train(model, optimizer, epochs=100):
  model = model.to(device=device)
  for epoch in range(epochs):
    for batch_idx, (xi, yi) in enumerate(train_loader):
      model.train()
      xi = xi.to(device=device, dtype=torch.float32)
      yi = yi.to(device=device, dtype=torch.long)

      scores = model(xi)
      cost = F.cross_entropy(input=scores, target=yi)
      optimizer.zero_grad()

      cost.backward()
      optimizer.step()
    train_acc = accuracy(model, train_loader)
    test_acc = accuracy(model, test_loader)
    print(f'Epoch: {epoch}, costo: {cost.item():.4f}, train_accuracy: {train_acc:.8f}, test_accuracy: {test_acc:.8f}')
    if train_acc == 1.0 and test_acc == 1.0: break
  # checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
  # save_checkpoint(checkpoint)


# * Hyperparameters
print(F'Initilizing Parameters')
learning_rate = 0.0001
num_epochs = 500


# * Initialize Network
print('Initializing Model')
model = LeNet().to(device)
# print(model)

# * Loss and Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('Begin Training')
train(model, optimizer, num_epochs)

model_path = 'LENET_500E.pth'
torch.save(model.state_dict(), model_path)