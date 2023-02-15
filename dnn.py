import numpy as np
import matplotlib.pyplot as plt
import torch

# load CIFAR-10 dataset
from torchvision import datasets, transforms

normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
transform = transforms.Compose([transforms.ToTensor(), normalize])  # normalize to (-1, 1)
trainset = datasets.CIFAR10('data', download=True, train=True, transform=transform)
testset = datasets.CIFAR10('data', download=True, train=False, transform=transform)

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


# test the models
# move the network to GPU
# resnet18 = resnet18.cuda()
# resnet34 = resnet34.cuda()
# resnet50 = resnet50.cuda()
# vgg16 = vgg16.cuda()
# mobileNet = mobileNet.cuda()
# output_resnet18 = test_model(resnet18, trainset)
# output_resnet34 = test_model(resnet34, trainset)
# output_resnet50 = test_model(resnet50, trainset)
# output_vgg16 = test_model(vgg16, trainset)
# output_mobileNet = test_model(mobileNet, trainset)
#
# print('ResNet18: ', output_resnet18.shape) # (50000, 512, 1, 1)
# print('ResNet34: ', output_resnet34.shape) # (50000, 512, 1, 1)
# print('ResNet50: ', output_resnet50.shape) # (50000, 2048, 1, 1)
# print('VGG16: ', output_vgg16.shape) # (50000, 512, 7, 7)
# print('MobileNet: ', output_mobileNet.shape) # (50000, 1280, 1, 1)
#
# # save the output of the models
# np.save('output_resnet18.npy', output_resnet18)
# np.save('output_resnet34.npy', output_resnet34)
# np.save('output_resnet50.npy', output_resnet50)
# np.save('output_vgg16.npy', output_vgg16)
# np.save('output_mobileNet.npy', output_mobileNet)

if __name__ == "__main__":
    models = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobileNet']
    for model in models:
        net = globals()[model].cuda()
        output, labels = test_model(net, trainset)
        np.save(f'train_{model}_data.npy', output)
        np.save(f'train_{model}_labels.npy', labels)

        output, labels = test_model(net, testset)
        np.save(f"test_{model}_data.npy", output)
        np.save(f"test_{model}_labels.npy", labels)
