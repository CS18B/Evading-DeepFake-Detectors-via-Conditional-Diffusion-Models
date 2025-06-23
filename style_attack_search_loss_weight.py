import os
import torch
from torch import optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from detector.xception_net import Xception_Net
from templates import *
from efficientnet_pytorch import EfficientNet
import lpips

# 设定设备
device = torch.device("cuda:3")

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载数据集
dir_path = '/home/coolboy/wwh/test_val/'
test_dataset = datasets.ImageFolder(dir_path, data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# 加载分类器模型
classifier = Xception_Net()
checkpoint = torch.load('/home/coolboy/wwh/diffusion-autoencoders-main/StyleGAN_weight/256_celebAHQ_detector/Xception_model_best_0.9781_epoch5.pt')
classifier.load_state_dict(checkpoint)
classifier = classifier.to(device)
classifier.eval()

# 加载自编码器模型
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'/home/coolboy/wwh/diffusion-autoencoders-main/checkpoints/ffhq256_autoenc/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# 定义LPIPS损失
lpips_loss = lpips.LPIPS(net='vgg').to(device)

# 定义交叉熵和MSE损失
entro_loss = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()

# 定义反向变换
reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # 再将[0, 1]映射到[-1, 1]
])

# 参数组合列表
param_combinations= [
    (1, 0, 0),    # 显著增加 loss_adv 的权重
    (0, 1, 0),    # 显著增加 l2_loss 的权重
    (0, 0, 1),
    (2, 1, 1),    # 显著增加 loss_adv 的权重
    (1, 2, 1),    # 显著增加 l2_loss 的权重
    (1, 1, 2),
    (0.5, 2, 1),    # 增加 l2_loss 的权重，减少 loss_adv 的权重
    (0.5, 1, 2),    # 增加 perceptual_loss 的权重，减少 loss_adv 的权重
    (1, 0.5, 2),    # 增加 perceptual_loss 的权重，减少 l2_loss 的权重
    (2, 0.5, 1),    # 增加 loss_adv 的权重，减少 l2_loss 的权重
    (2, 1, 0.5),    # 增加 loss_adv 的权重，减少 perceptual_loss 的权重
    (1, 2, 0.5),    # 增加 l2_loss 的权重，减少 perceptual_loss 的权重
    (0.5, 1, 1), 
    (1, 0.5, 1), 
    (1, 1, 0.5), 
    (0.5, 0.5, 1), 
    (1, 0.5, 0.5), 
    (0.5, 1, 0.5),
    (0.1, 2, 0.1), 
    (0.5, 2, 0.5),
    (0.2, 3, 0.2), 
    (0.1, 5, 0.1), 
    (0.5, 1.5, 0.5),
    (0.3, 2, 0.3),
    (0.4, 2.5, 0.4),
    (0.2, 2, 0.1),
    (0.1, 3, 0.5),
    (0.1, 4, 0.1),
    (0.3, 3, 0.3),
    (0.2, 4, 0.2),
    (0.1, 10, 0.1),
]

def save_adv_images(adv_images, file_path):
    pil_image = to_pil_image(adv_images.squeeze(0))
    pil_image.save(file_path)

# 对抗攻击函数
def adversarial_attack(model, classifier, images, labels, k, weight_adv, weight_l2, weight_perceptual, current_combination):
    style_embeddings = model.encode(images)
    xT = model.encode_stochastic(images, style_embeddings, T=100)
    latent = style_embeddings.clone().detach()

    # 创建优化器，将latent作为需要更新的参数
    optimizer = optim.AdamW([latent], lr=0.003)
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    break_step = 20

    for i in range(k):
        latent.requires_grad = True
        optimizer.zero_grad()  # 清除之前的梯度
        xT_adv = model.render(xT, latent, T=10)
        # print("xT_adv range during attack:", xT_adv.min().item(), xT_adv.max().item())
        xT_adv_mapped = reverse_transform(xT_adv)
        # print("xT_adv_mapped range during attack:", xT_adv_mapped.min().item(), xT_adv_mapped.max().item())
        output = classifier(xT_adv_mapped).float()
        labels = labels.long()
        
        loss_adv = entro_loss(output, labels)
        l2_loss = mse_loss(xT_adv_mapped, images)
        perceptual_loss = lpips_loss(xT_adv_mapped, images)
        
        total_loss = weight_adv * loss_adv + weight_l2 * l2_loss + weight_perceptual * perceptual_loss

        total_loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        
        pred = classifier(xT_adv_mapped)
        pred_label = torch.argmax(pred, 1).detach()
        print(f'Params {current_combination} - Current image label {pred_label} iteration {i} loss {total_loss.item()}')
        
        if pred_label == 1:
            print(f'Breaking at iteration {i} with params {current_combination} as the predicted label is 1')
            break_step = i
            break

    return xT_adv, break_step


# 保存图片的函数


# 进行对抗攻击并保存结果
# 在主循环中调用 adversarial_attack
for iter, (images, labels) in enumerate(test_loader, start=1):
    images, ori_labels = images.to(device), labels.to(device)
    labels = torch.ones_like(ori_labels, dtype=torch.long)  # 假设我们想要的是类别1

    for weights in param_combinations:
        weight_adv, weight_l2, weight_perceptual = weights
        xT_adv, break_step = adversarial_attack(model, classifier, images, labels, 20, weight_adv, weight_l2, weight_perceptual, weights)
        xT_adv_mapped = reverse_transform(xT_adv)
        
        # 保存对抗样本图片
        save_path = f'/home/coolboy/wwh/test_val/Real/{break_step}_adv_{weight_adv}_{weight_l2}_{weight_perceptual}.png'
        save_adv_images(xT_adv, save_path)



    if iter == 1:  # 如果只想处理一个批次，取消注释这行
        break
