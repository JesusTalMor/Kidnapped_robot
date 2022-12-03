import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LENET_model = LeNet().to(device=device)
LENET_model.load_state_dict(torch.load('/LENET_300E_ROT.pth.pth'))

LENET_model.eval()

def sample_image():
  rnd_idx = np.random.randint(low=295, high=500)
  img_path = f'images/case_{rnd_idx}.png'
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  image = cv2.resize(image, (128, 128))
  plt.imshow(image, cmap='gray')
  plt.show()
  return torch.tensor(image).to(device=device, dtype=torch.float).view(1, 1, 128, 128)

for x in range(100):
  image = sample_image()
  _, pred = LENET_model(image).max(1)
  decision = 'Kidnaped' if pred[0] == 1 else 'Normal'
  print(decision)