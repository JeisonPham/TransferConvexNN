import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from torchvision import datasets, transforms


# load flower dataset
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
        if len(num) < 4:
            diff = 4 - len(num)
            for i in range(diff):
                num = '0' + num
        path = "flower_data/jpg/image_" + num + ".jpg"
        picture = Image.open(path)
        label = img_id // 80
        picture = self.transforms(picture)
        return picture, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    transforms.Resize((224, 224))
]
)


index = np.random.permutation(np.arange(1360))
train_index = index[:1020]
test_index = index[1020:]
trainset = dataset(train=True, data_index=train_index, transforms=transform)
testset = dataset(train=False, data_index=test_index, transforms=transform)


# define the network
# load resnet18
from torchvision import models

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
mobileNet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# freeze the parameters
for param in resnet18.parameters():
    param.requires_grad = False
for param in resnet34.parameters():
    param.requires_grad = False
for param in resnet50.parameters():
    param.requires_grad = False
for param in vgg16.parameters():
    param.requires_grad = False
for param in mobileNet.parameters():
    param.requires_grad = False

# remove the last average pooling layer
resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
resnet34 = torch.nn.Sequential(*(list(resnet34.children())[:-1]))
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
vgg16 = torch.nn.Sequential(*(list(vgg16.children())[:-1]))
mobileNet = torch.nn.Sequential(*(list(mobileNet.children())[:-1]))

# Write all the layers in the network to file
with open('model_info.txt', 'w') as f:
    f.write('ResNet18')
    f.write(str(resnet18))
    f.write('\n\n ResNet34')
    f.write(str(resnet34))
    f.write('\n\n ResNet50')
    f.write(str(resnet50))
    f.write('\n\n VGG16')
    f.write(str(vgg16))
    f.write('\n MobileNet')
    f.write(str(mobileNet))

# test each of the models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def test_model(model, trainset):
    model.eval()
    testloader = DataLoader(trainset, batch_size=128, shuffle=True)
    output_total = []
    labels_total = []
    for images, labels in tqdm(testloader):
        images, labels = images.cuda(), labels.cuda()
        output = model(images)
        output = output.cpu().detach().numpy()
        output_total.append(output)
        labels_total.append(labels.cpu().detach().numpy())
    output_total = np.concatenate(output_total, axis=0)
    labels = np.concatenate(labels_total, axis=0)

    return output_total, labels


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pool', type=int, default=0)
    max_pool = parser.parse_args().max_pool


    models = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobileNet']
    for model in models:
        net = globals()[model].cuda()
        train_output, train_labels = test_model(net, trainset)

        test_output, test_labels = test_model(net, testset)


        # bilinear interpolation seems to be better used for dl model input, not output
        # do max pooling to reduce the dimension
        if max_pool == 1:
            train_output = train_output.reshape(train_output.shape[0], -1)
            test_output = test_output.reshape(test_output.shape[0], -1)
            if model == 'resnet18' or model == 'resnet34': # (50000, 512, 1, 1)
                train_output = torch.nn.functional.max_pool1d(torch.from_numpy(train_output), kernel_size=2)
                test_output = torch.nn.functional.max_pool1d(torch.from_numpy(test_output), kernel_size=2)
            elif model == 'resnet50': # (50000, 2048, 1, 1)
                train_output = torch.nn.functional.max_pool1d(torch.from_numpy(train_output), kernel_size=8)
                test_output = torch.nn.functional.max_pool1d(torch.from_numpy(test_output), kernel_size=8)
            elif model == 'vgg16': # (50000, 512, 7, 7)
                train_output = torch.nn.functional.max_pool1d(torch.from_numpy(train_output), kernel_size=98)
                test_output = torch.nn.functional.max_pool1d(torch.from_numpy(test_output), kernel_size=98)
            elif model == 'mobileNet': # (50000, 1280, 1, 1)
                train_output = torch.nn.functional.max_pool1d(torch.from_numpy(train_output), kernel_size=5)
                test_output = torch.nn.functional.max_pool1d(torch.from_numpy(test_output), kernel_size=5)

            # convert it back to 50000 x 256 x 1 x 1
            train_output = train_output.numpy()
            test_output = test_output.numpy()
            train_output = train_output.reshape(1020, 256, 1, 1)
            test_output = test_output.reshape((1360-1020), 256, 1, 1)

        np.save(f'train_{model}_data.npy', train_output)
        np.save(f'train_{model}_labels.npy', train_labels)

        np.save(f"test_{model}_data.npy", test_output)
        np.save(f"test_{model}_labels.npy", test_labels)


