

import torch 
import torchvision 
import numpy as np

from google.colab import drive
drive.mount("/content/drive")

!unzip /content/drive/MyDrive/flower_dataset.zip

from PIL import Image

data = Image.open("/content/jpg/image_0001.jpg")

data

index = np.random.permutation(np.arange(1360))

index.shape

train_index = index[:1020]

train_index.shape

test_index = index[1020:]

test_index.shape

class dataset(torch.utils.data.Dataset):
  def __init__(self, train, data_index, transforms):
    super().__init__()
    self.train = train
    self.data_index = data_index
    self.transforms = transforms
  def __len__(self):
    return len(self.data_index)
  def __getitem__(self, index):
    img_id = self.data_index[index]
    num = str(self.data_index[index]+1)
    if len(num)<4:
      diff = 4-len(num)
      for i in range(diff):
        num = "0"+num
    path = "/content/jpg/image_"+num+".jpg"
    picture = Image.open(path)
    label = img_id // 80
    picture = self.transforms(picture)
    return picture, label

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
]
)

train_dataset = dataset(train=True, data_index=train_index, transforms=transform)
test_dataset = dataset(train=False, data_index=test_index, transforms=transform)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=False)



model = torch.nn.Sequential(model, torch.nn.Linear(1000,17))



model

device = torch.device("cuda")
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
loss_function = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

from tqdm import tqdm
model.train()
epochs = 30
for epoch in tqdm(range(epochs)):
  total_acc = 0
  total_loss = 0
  for i, (data, label) in enumerate(train_dataloader):
    data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, label)
    loss.backward()
    optimizer.step()
    pred = torch.max(output, dim=1)[1]
    acc = torch.sum(pred==label)
    total_acc+=acc
    total_loss +=loss.item()
  lr_scheduler.step()
  print("epoch: {}\taccuracy: {}\t loss: {}".format(epoch+1, total_acc/1020, total_loss/len(train_dataloader)))

total_acc = 0
for i, (data, label) in enumerate(test_dataloader):
  model.eval()
  data, label = data.to(device), label.to(device)
  output = model(data)
  pred = torch.max(output, 1)[1]
  acc = torch.sum(pred == label)
  total_acc+=acc

print(total_acc/len(test_dataloader))