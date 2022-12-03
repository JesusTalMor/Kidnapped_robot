import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from customDataset import kidnappedDataset

# Load Data
my_transform = transforms.Compose([
  transforms.ToPILImage(), 
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomRotation(degrees=90),
  transforms.ToTensor()
])

dataset = kidnappedDataset(csv_file= 'kidnaped_dataset.csv', root_dir='images', transform=my_transform)

for index, (img, label) in enumerate(dataset):
  if index == 3: break
  save_image(img, f'img{index}.png')
