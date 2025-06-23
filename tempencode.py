from templates import *

device = 'cuda:1'
conf = ffhq128_autoenc_72M()
# conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'/home/coolboy/wwh/diffusion-autoencoders-main/checkpoints/ffhq128_autoenc_72M/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

def process_and_save_conditions(dataset_path, save_path, model, device):
    # 加载数据
    dataset = ImageDataset(dataset_path, image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

    # 保存每张图像的编码到列表中
    encoded_conditions = []
    total_images = len(dataset)

    for idx in range(total_images):
        if idx % 1000 == 0:
            print(idx)
        image_batch = dataset[idx]['img'][None]
        condition = model.encode(image_batch.to(device))
        encoded_conditions.append(condition)

    torch.save(encoded_conditions, save_path)

# 处理伪造数据
process_and_save_conditions('/home/coolboy/wwh/Data/Fake', 'weight/fake_conds180k.pth', model, device)

# 处理真实数据
process_and_save_conditions('/home/coolboy/wwh/Data/Real', 'weight/real_conds180k.pth', model, device)
