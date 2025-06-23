from torch import optim
from detector.xception_net import Xception_Net
import torch
from torchvision import transforms, datasets
from templates import *
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda:3")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 

tensor_mu = torch.tensor(mean).view(1, 3, 1, 1)
tensor_std = torch.tensor(std).view(1, 3, 1, 1)


data_transforms = {
    'train': transforms.Compose([
        # Aug(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


dir_path = '/home/coolboy/wwh/diffusion-autoencoders-main/imgs_val'
batch_size = 1
test_dataset = datasets.ImageFolder(dir_path, data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataset_size = len(test_dataset)
print('size: {}'.format(dataset_size))

classifier = Xception_Net()
check = torch.load('/home/coolboy/wwh/diffusion-autoencoders-main/weight/128_weight/Xcep_model_best_0.9990_epoch7.pt')
classifier.load_state_dict(check)
classifier = classifier.to(device)
classifier.eval()
correct, TP, TN, FP, FN = 0, 0, 0, 0, 0


cross_entro = torch.nn.CrossEntropyLoss()
conf = ffhq128_autoenc_72M()
model = LitModel(conf)
state = torch.load(f'/home/coolboy/wwh/diffusion-autoencoders-main/checkpoints/ffhq128_autoenc_72M/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.train()
model.ema_model.to(device)

torch.autograd.set_detect_anomaly(True)

for images, labels in test_loader:
    
    images = images.to(device)
    labels = labels.to(device)
    style_embeddings = model.encode(images)
    style_embeddings.requires_grad_(True)
    # print(style_embeddings.grad)

    xT = model.encode_stochastic(images, style_embeddings, T=10)
    optimizer = optim.AdamW([style_embeddings], lr=1e-2)
    
    # 检查优化器的状态
    for _ in range(10):
        style_embeddings.requires_grad_(True)
        # print(style_embeddings.grad)
        adversarial_image = model.render(xT, style_embeddings, T=10)
        pred = classifier(adversarial_image)
        attack_loss = cross_entro(pred, labels)
        loss = attack_loss

        
        
        print(f"attack_loss: {attack_loss.item():.5f} ")
        print(f"loss: {loss.item():.5f}")
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # print('grad grad', style_embeddings.grad)
        optimizer.step()
        classifier.zero_grad()
        style_embeddings=style_embeddings.detach()

    pred = classifier(adversarial_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == labels).sum().item() / len(labels)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))
    

    logit = torch.nn.Softmax(dim=1)(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", labels, logit[0, labels[0]])