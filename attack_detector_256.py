from templates import *
import matplotlib.pyplot as plt
device = 'cuda:0'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'/home/coolboy/wwh/diffusion-autoencoders-main/checkpoints/ffhq256_autoenc/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data1 = ImageDataset('/home/coolboy/wwh/diffusion-autoencoders-main/temp', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch1 = data1[1]['img'][None]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

cond1 = model.encode(batch1.to(device))
xT_1 = model.encode_stochastic(batch1.to(device), cond1, T=250).to(device)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch1 + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT_1[0].permute(1, 2, 0).cpu())

# save_image(ori[0], f"/home/coolboy/wwh/diffusion-autoencoders-main/a_ori.png")
# save_image(xT_1[0], f"/home/coolboy/wwh/diffusion-autoencoders-main/a_noise.png")


# pred = model.render(xT_1, combine_vector, T=100)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ori = (batch1 + 1) / 2
# ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
# ax[1].imshow(pred[0].permute(1, 2, 0).cpu())
# # 保存整个plt输出
# fig.savefig("/home/coolboy/wwh/diffusion-autoencoders-main/imgs_render/Fake/complete_output_image.png")

# 仅保存pred图像
# save_image(pred[0], f"/home/coolboy/wwh/diffusion-autoencoders-main/imgs_render/Real/pred1.png")


pred = model.render(xT_1, cond1, T=35)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch1 + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(pred[0].permute(1, 2, 0).cpu())