import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from detector.xception_net import Xception_Net
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

device = torch.device("cuda:3")
data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 如果必要，调整大小以确保图像尺寸一致
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # 只是将PIL图像转为Tensor，不进行归一化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

savefig_path = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_images/FGSM_png/fake'
dir_path = '/home/coolboy/wwh/stylegan_wb_test/test_val'
batch_size = 20
test_dataset = datasets.ImageFolder(dir_path, data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataset_size = len(test_dataset)
print('size: {}'.format(dataset_size))
print('Class to index mapping:', test_dataset.class_to_idx)

# Xception
weightpathA = '/home/coolboy/wwh/diffusion-autoencoders-main/256_StyleGAN_weight/256_celebAHQ_detector/Xception_model_best_0.9781_epoch5.pt'
# Efficient
weightpathB = '/home/coolboy/wwh/diffusion-autoencoders-main/256_StyleGAN_weight/256_celebAHQ_detector/Effnet_model_best_0.9900_epoch2.pt'
# ResNet
weightpathC = '/home/coolboy/wwh/diffusion-autoencoders-main/256_StyleGAN_weight/256_celebAHQ_detector/Resnet_model_best_0.5515_epoch8.pt'
# MobileNet
weightpathD = '/home/coolboy/wwh/diffusion-autoencoders-main/256_StyleGAN_weight/256_celebAHQ_detector/Mobile_model_best_0.9899_epoch13.pt'

modelA = Xception_Net().to(device)
checkA = torch.load(weightpathA)
modelA.load_state_dict(checkA)
modelA.eval()


modelB = EfficientNet.from_name('efficientnet-b7').to(device)

new_state_dict = OrderedDict()
checkpointB = torch.load(weightpathB)
for k, v in checkpointB.items():
    # 移除'module.'的前缀
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

# 用新的state_dict加载模型
modelB.load_state_dict(new_state_dict)
# checkB = torch.load(weightpathB)
# modelB.load_state_dict(checkB)
modelB.eval()


modelD = torchvision.models.mobilenet_v3_large(pretrained=False)
modelD.classifier[3] = nn.Linear(modelD.classifier[3].in_features, 2)
checkD = torch.load(weightpathD)
modelD.load_state_dict(checkD)
modelD = modelD.to(device)
modelD.eval()


# ...（前面的代码保持不变）

# 测试循环

for batch_idx, (images, labels) in enumerate(test_loader):
    if batch_idx > 1: break
    inputs = images.to(device)
    labels = labels.to(device)

    # 获取当前批次的文件路径
    paths = [test_dataset.samples[i][0] for i in range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, dataset_size))]

    # 不计算梯度，因为我们不在训练模式
    with torch.no_grad():
        # 对每个模型进行前向传播
        outputA = modelA(inputs)
        outputB = modelB(inputs)
        outputD = modelD(inputs)

        # 计算预测结果
        _, predictionA = torch.max(outputA, 1)
        _, predictionB = torch.max(outputB, 1)
        _, predictionD = torch.max(outputD, 1)

        # 遍历批次中的每个样本
        for idx, path in enumerate(paths):
            # 获取文件名
            filename = path.split('/')[-1]
            # 打印文件名和预测结果
            print(f'Filename: {filename}')
            print(f'Model A prediction: {predictionA[idx].item()}')
            print(f'Model B prediction: {predictionB[idx].item()}')
            print(f'Model D prediction: {predictionD[idx].item()}')
            # if predictionA[idx].item() == 1 and predictionB[idx].item() == 1 and predictionD[idx].item() == 1:
            #     print('********************************************************************************')
            print(f'Model A logit: {outputA[idx]}')
            print(f'Model B logit: {outputB[idx][:2]}')
            print(f'Model C logit: {outputD[idx]}')
            print()

# # 测试循环
# for images, labels in test_loader:
#     inputs = images.to(device)
#     labels = labels.to(device)

#     # 不计算梯度，因为我们不在训练模式
#     with torch.no_grad():
#         # 对每个模型进行前向传播
#         outputA = modelA(inputs)
#         outputB = modelB(inputs)
#         outputD = modelD(inputs)

#         # 计算预测结果
#         _, predictionA = torch.max(outputA, 1)
#         _, predictionB = torch.max(outputB, 1)
#         _, predictionD = torch.max(outputD, 1)

#         # 打印输出
#         print('Model A output:', outputA)
#         print('Model A prediction:', predictionA)
#         print('Model B output:', outputB)
#         print('Model B prediction:', predictionB)
#         print('Model D output:', outputD)
#         print('Model D prediction:', predictionD)




    
# print('Final Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(precision, recall, f1_score))

# str1 = 'model {} Final Prediction Acc: {}%'.format(str(modelA)[:10], acc * 100)
# str2 = 'Final Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(precision, recall, f1_score)
# str3 = 'Final True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(TP, TN, FP, FN)


# with open('result.txt', 'a') as file:
#     file.write('\n')
#     file.write(str1)
#     file.write('\n')
#     file.write(str2)
#     file.write('\n')
#     file.write(str3)
#     file.write('\n')

# def denorm(x, mean: Tensor, std: Tensor):
#     """Convert the range from [-1, 1] to [0, 1]."""
#     # out = (x + 1) / 2
#     # return out.clamp_(0, 1)
#     return x * std + mean

# img = inputs.cpu().detach()[0]
# img = denorm(img, torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1))

# def norm(x, mean, std):
#     return (x - mean) / std

# def fgsm_attack_saveimg(model, loss, images, labels, eps=8/255):

    # images = images.to(device)
    # labels = labels.to(device)
    # images.requires_grad = True
    # outputs = model(images)
    # model.zero_grad()
    # cost = loss(outputs, labels).to(device)
    # cost.backward()
    # # Denormalize the input images
    # images_denorm = denorm(images.cpu(), tensor_mu, tensor_std).to(device)

    # # Add the perturbation and clip the values to [0, 1]
    # attack_images_denorm = images_denorm + eps * images.grad.sign().to(device)
    # attack_images_denorm = torch.clamp(attack_images_denorm, 0, 1)

    # # Normalize the adversarial images back to the expected input range for the model
    # attack_images = norm(attack_images_denorm.cpu(), tensor_mu, tensor_std).to(device)

    # return attack_images

# def pgd_attack_saveimg(model, images, labels, eps=8/255, alpha=1/255, iters=10):
#     images = images.to(device)
#     labels = labels.to(device)
#     loss = nn.CrossEntropyLoss()

#     ori_images = images.data

#     # Denormalize the input images
#     ori_images_denorm = denorm(ori_images.cpu(), tensor_mu, tensor_std).to(device)

#     for i in range(iters):
#         images.requires_grad = True
#         outputs = model(images)

#         model.zero_grad()
#         cost = loss(outputs, labels).to(device)
#         cost.backward()

#         # Calculate the perturbation (eta) and clamp it to the specified range
#         # pertubation = alpha * images.grad.sign()
#         pertubation = torch.clamp(alpha * images.grad.sign(), min=-eps, max=eps)
#         adv_images_denorm = torch.clamp(ori_images_denorm + pertubation, min=0, max=1).detach_()

#         # Normalize the adversarial images back to the expected input range for the model
#         images = norm(adv_images_denorm.cpu(), tensor_mu, tensor_std).to(device)

#     return images