import os
import torch
from PIL import Image
import pyiqa
from torchvision import transforms as T
import csv
import numpy as np

device = torch.device('cuda:1')
torch.cuda.set_device(device)

transform = T.Compose([
    T.ToTensor()
])

csv_filename = f'/home/coolboy/wwh/xiaorong_visual_FFHQ.csv'
csv_header = ['Subfolder', 'BRISQUE', 'DBCNN', 'FID', 'clipiqa', 'clipiqa+', 'clipiqa+_vitL14_512']

with open(csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

iqa_loss_brisque = pyiqa.create_metric('brisque').to(device)
iqa_dbcnn = pyiqa.create_metric('dbcnn').to(device)
iqa_clip_iqa = pyiqa.create_metric('clipiqa').to(device)
iqa_clip_iqa_plus = pyiqa.create_metric('clipiqa+').to(device)
iqa_clip_iqa_plus_vitL14_512 = pyiqa.create_metric('clipiqa+_vitL14_512').to(device)

def process_folder(folder_to_compare, base_folder):
    subfolders = os.listdir(folder_to_compare)

    # 检查是否只有一个子目录
    if len(subfolders) == 1 and os.path.isdir(os.path.join(folder_to_compare, subfolders[0])):
        # 如果只有一个子目录，则直接在该目录中处理文件
        subfolders = [""]
    
    for subfolder in subfolders:
        fake_folder_path = os.path.join(folder_to_compare, subfolder, 'Fake')
        print(f'processing {fake_folder_path}')

        if not os.path.isdir(fake_folder_path):
            print(f'{fake_folder_path} missing!')
            continue

        brisque_values, dbcnn_values, clip_iqa_values = [], [], []
        clip_iqa_plus_values, clip_iqa_plus_vitL14_512_values = [], []
        
        file_count = 0 
        for file in sorted(os.listdir(fake_folder_path)):
            file_count += 1
            if file_count % 100 == 0:  # 每处理100个文件打印一次进度
                print(f"Processed {file_count} files.")
            file_path = os.path.join(fake_folder_path, file)

            img_inv = Image.open(file_path).convert('RGB')

            brisque_value = iqa_loss_brisque(img_inv).item()
            dbcnn_value = iqa_dbcnn(img_inv).item()
            clip_iqa_value = iqa_clip_iqa(img_inv).item()
            clip_iqa_plus_value = iqa_clip_iqa_plus(img_inv).item()
            clip_iqa_plus_vitL14_512_value = iqa_clip_iqa_plus_vitL14_512(img_inv).item()

            brisque_values.append(brisque_value)
            dbcnn_values.append(dbcnn_value)
            clip_iqa_values.append(clip_iqa_value)
            clip_iqa_plus_values.append(clip_iqa_plus_value)
            clip_iqa_plus_vitL14_512_values.append(clip_iqa_plus_vitL14_512_value)

        brisque_mean = np.mean(brisque_values)
        dbcnn_mean = np.mean(dbcnn_values)
        clip_iqa_mean = np.mean(clip_iqa_values)
        clip_iqa_plus_mean = np.mean(clip_iqa_plus_values)
        clip_iqa_plus_vitL14_512_mean = np.mean(clip_iqa_plus_vitL14_512_values)

        fid_metric = pyiqa.create_metric('fid').to(device)
        fid_score = fid_metric(base_folder, fake_folder_path).item()

        # 将结果保存到CSV中
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([subfolder, brisque_mean, dbcnn_mean, fid_score, clip_iqa_mean, clip_iqa_plus_mean, clip_iqa_plus_vitL14_512_mean])

base_folder_2 = '/home/coolboy/wwh/256-5-attribute-Diff-t30/test/Real/FFHQ'
folders_to_compare_2 = '/home/coolboy/wwh/instruct-pix2pix-main/adversarial'

process_folder(folders_to_compare_2, base_folder_2)

print(f"指标计算完成并保存到{csv_filename}")
