import os
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from PIL import Image
import lpips
import pyiqa
import torch.nn.functional as F
from torchvision import transforms as T
import csv
# from pytorch_msssim import ms_ssim

device = torch.device('cuda:1')
torch.cuda.set_device(device)

# Functions to compute the metrics
def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def l2_norm(y_true, y_pred):
    n = y_true.shape[0] * y_true.shape[1] * y_true.shape[2]
    return torch.sqrt(torch.sum((y_true - y_pred) ** 2) / n)

def compute_metrics(y_true, y_pred):
    mse_value = mse(y_true, y_pred).item()
    l2_norm_value = l2_norm(y_true, y_pred).item()
    return mse_value, l2_norm_value


transform = T.Compose([
    T.ToTensor()
])


csv_filename = f'/home/coolboy/wwh/xiaorong_visual_CelebAHQ.csv'
csv_header = ['Subfolder', 'MSE', 'LPIPS', 'PSNR', 'SSIM', 'L2_norm', 'BRISQUE', 'NIMA', 'DBCNN', 'FID']


with open(csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)


iqa_loss_lpips = pyiqa.create_metric('lpips-vgg').to(device)
iqa_loss_ssim = pyiqa.create_metric('ssim').to(device)
iqa_loss_psnr = pyiqa.create_metric('psnr').to(device)
iqa_loss_brisque = pyiqa.create_metric('brisque').to(device)
iqa_nima = pyiqa.create_metric('nima').to(device)
iqa_dbcnn = pyiqa.create_metric('dbcnn').to(device)

def process_folder(folder_to_compare, base_folder):

    subfolders = os.listdir(folder_to_compare)

    # 检查是否只有一个子目录
    if len(subfolders) == 1 and os.path.isdir(os.path.join(folder_to_compare, subfolders[0])):
        # 如果只有一个子目录，则直接在该目录中处理文件
        subfolders = [""]
    
    for subfolder in subfolders:
        fake_folder_path = os.path.join(folder_to_compare, subfolder, 'fake')
        print(f'processing {fake_folder_path}')

        if not os.path.isdir(fake_folder_path):
            print(f'{fake_folder_path} missing!')
            continue

        mse_values, ssim_values, lpips_values, psnr_values, l2_norm_values, brisque_values, nima_values, dbcnn_values = [], [], [], [], [], [], [], []
        
        file_count = 0 
        for file in sorted(os.listdir(fake_folder_path), key=lambda x: int(x.split('.')[0])):
            file_count += 1
            if file_count % 100 == 0:  # 每处理100个文件打印一次进度
                print(f"Processed {file_count} files.")
            file_path = os.path.join(fake_folder_path, file)
            file_number = int(file.split('.')[0])  # 提取文件编号
            base_file_name = f'{file_number:04}.png'  # 将文件编号转换为四位数格式
            base_file_path = os.path.join(base_folder, base_file_name)

            if not (os.path.exists(base_file_path) and os.path.exists(os.path.join(fake_folder_path, file))):
                print(f"File not found: {base_file_path} or {file}")
                continue

            img_ori = Image.open(base_file_path).convert('RGB')
            img_inv = Image.open(file_path).convert('RGB')

            img_ori_tensor = transform(img_ori).unsqueeze(0).to(device)
            img_inv_tensor = transform(img_inv).unsqueeze(0).to(device)

            lpips_value = iqa_loss_lpips(img_ori, img_inv).item()
            ssim_value = iqa_loss_ssim(img_ori, img_inv).item()       
            psnr_value = iqa_loss_psnr(img_ori, img_inv).item()    
            brisque_value = iqa_loss_brisque(img_inv).item()
            nima_value = iqa_nima(img_inv).item()
            dbcnn_value = iqa_dbcnn(img_inv).item()
            mse_value, l2_norm_value = compute_metrics(img_ori_tensor, img_inv_tensor)

            mse_values.append(mse_value)
            lpips_values.append(lpips_value)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            l2_norm_values.append(l2_norm_value)
            brisque_values.append(brisque_value)
            nima_values.append(nima_value)
            dbcnn_values.append(dbcnn_value)

        mse_mean = np.mean(mse_values)
        lpips_mean = np.mean(lpips_values)
        psnr_mean = np.mean(psnr_values)
        ssim_mean = np.mean(ssim_values)
        l2_norm_mean = np.mean(l2_norm_values)
        brisque_mean = np.mean(brisque_values)
        nima_mean = np.mean(nima_values)
        dbcnn_mean = np.mean(dbcnn_values)

        fid_metric = pyiqa.create_metric('fid').to(device)
        fid_score = fid_metric(base_folder, fake_folder_path).item()

        # 将结果保存到CSV中
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([subfolder, mse_mean, lpips_mean, psnr_mean, ssim_mean, l2_norm_mean, brisque_mean, nima_mean, dbcnn_mean, fid_score])


# base_folder_1 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_CelebAHQ_stargan/Fake_to_generate_1500/Fake'
base_folder_2 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_Diff/Style_Attack_lr_test/original'
# folders_to_compare_1 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_CelebAHQ_stargan/noise_adv'
# folders_to_compare_1 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_CelebAHQ_stargan/SA'
folders_to_compare_2 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_Diff/Style_Attack_lr_test'
# folders_to_compare_3 = '/home/coolboy/wwh/statattack/CelebAHQ'
# folders_to_compare_1 = '/home/coolboy/wwh/diffusion-autoencoders-main/256_ddim_CelebAHQ_stargan/SA_srm_bad/srm_iter30_1_1_0.5'
# process_folder(folders_to_compare_1, base_folder_1)
process_folder(folders_to_compare_2, base_folder_2)
# process_folder(folders_to_compare_2, base_folder)
# process_folder(folders_to_compare_3, base_folder)

print(f"指标计算完成并保存到{csv_filename}")