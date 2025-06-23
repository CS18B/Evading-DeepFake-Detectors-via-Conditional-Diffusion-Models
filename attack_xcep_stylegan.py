import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from detector.xception_origin import Xception
from detector.mesonet import Meso4
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import Tensor
import torchattacks
import numpy as np
import utils
from torchvision.utils import save_image
from collections import OrderedDict
from detector.SPAD import xception_net
from detector.model_core import Two_Stream_Net

device = torch.device("cuda:0")
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
mean_tensor = torch.tensor(mean).view(3, 1, 1)
std_tensor = torch.tensor(std).view(3, 1, 1)

data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 如果必要，调整大小以确保图像尺寸一致
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # 只是将PIL图像转为Tensor，不进行归一化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


dir_path = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_FFHQ_stylegan/Fake_to_generate_1500'
batch_size = 128
test_dataset = datasets.ImageFolder(dir_path, data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
dataset_size = len(test_dataset)
print('size: {}'.format(dataset_size))
print('Class to index mapping:', test_dataset.class_to_idx)


weightpathA = '/home/coolboy/wwh/SPAD-main/net-best_celebahq.ckpt'

weightpathB = '/home/coolboy/wwh/diffusion-autoencoders-main/256_weight_FFHQ_stylegan/detector_weight/Xception_0.9998_0.9845_epoch8.pt'


# modelA = xception_net().to(device)
# checkA = torch.load(weightpathA)
# modelA.load_state_dict(checkA)
# modelA.eval()


classifier = Xception().to(device)
state_dict = torch.load(weightpathB)
classifier.load_state_dict(state_dict, False)
classifier.eval()


correct = 0
criterion = nn.CrossEntropyLoss()
TP, TN, FP, FN = 0, 0, 0, 0


model_attack = classifier
attack = torchattacks.FGSM(model_attack, eps=8/255)
# attack = torchattacks.PGD(model_attack, eps=8/255, steps=40)
# attack = torchattacks.MIFGSM(model_attack, eps=8/255)
# attack = torchattacks.VMIFGSM(model_attack, eps=8/255)
attack.set_normalization_used(mean=mean, std=std)

def denorm(x, mean: Tensor, std: Tensor):
    """Convert the range from [-1, 1] to [0, 1]."""
    return x * std + mean


interrupt = 15
i = 0


for images, labels in test_loader:
    i += 1
    if i > interrupt: break
    adv_images = attack(images, labels).to(device)
    # adv_images = images.to(device)
    # adv_images = adv_images.cpu().detach()[0]
    # adv_images = denorm(adv_images, torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1))
    # save_image(adv_images, '{}/{}.png'.format(savefig_path, i))
    
    labels = labels.to(device)
    with torch.no_grad():
        output = model_attack(adv_images).to(device).float()
        # output = modelC(adv_images).to(device).float()
    _, prediction = torch.max(output, 1)
    correct += torch.sum((prediction.detach()) == labels.detach().to(torch.float32))
    TP += torch.sum((prediction == 1) & (labels == 1))
    TN += torch.sum((prediction == 0) & (labels == 0))
    FP += torch.sum((prediction == 1) & (labels == 0))
    FN += torch.sum((prediction == 0) & (labels == 1))

    precision = TP.float() / (TP + FP).float()
    recall = TP.float() / (TP + FN).float()
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(TP, TN, FP, FN))
    print()


print('Pre-Acc {:.4f}%， After-Acc: {:.4f}%, ASR: {:.4f}%'.format(FN/(TN+FP)*100, TN/(TN+FP)*100 ,FP/(TN+FP)*100))
print('Final True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(TP, TN, FP, FN))
print('path: {}'.format(dir_path[-8:]))

