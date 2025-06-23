from torch import optim
from detector.xception_net import Xception_Net
import torch
from torchvision import transforms, datasets
from templates import *
from efficientnet_pytorch import EfficientNet
from torchvision.utils import save_image

device = torch.device("cuda:3")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 
mean_tensor = torch.tensor(mean).view(3, 1, 1).to(device)
std_tensor = torch.tensor(std).view(3, 1, 1).to(device)

data_transforms = {
    'train': transforms.Compose([
        # Aug(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


dir_path = '/home/coolboy/wwh/test_val/'
batch_size = 1
test_dataset = datasets.ImageFolder(dir_path, data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataset_size = len(test_dataset)
print('size: {}'.format(dataset_size))



cross_entro = torch.nn.CrossEntropyLoss()
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'/home/coolboy/wwh/diffusion-autoencoders-main/checkpoints/ffhq256_autoenc/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.to(device)

# torch.autograd.set_detect_anomaly(True)
def denorm(x, mean, std):
    """Convert the range from [-1, 1] to [0, 1]."""
    return x * std + mean

iter = 0
interrupt = 1
for images, labels in test_loader:
    iter += 1
    if iter > interrupt: break
    images = images.to(device)
    labels = labels.to(device)
    style_embeddings = model.encode(images)
    print(style_embeddings.shape)
    xT = model.encode_stochastic(images, style_embeddings, T=10)
    # latent = style_embeddings.clone().detach()
    xT_adv = model.render(xT, style_embeddings, T=10)
    
    xT_adv = xT.squeeze(0)
    # adv_images = denorm(xT_adv, mean_tensor, std_tensor)
    save_image(xT_adv.cpu(), '{}/{}.png'.format('/home/coolboy/wwh/test_val/Real', iter))
    
