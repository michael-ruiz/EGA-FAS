import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
# from data_process.augmentation_1 import *
from data_process.augmentation import *
from data_process.augmentation import strong_color_augumentor
from skimage.color import rgb2ycbcr


RESIZE_SIZE = 128

def Data_Cut(config):
    """
    目前多模态数据集包括CASIA-SURF和WMCA
    CASIA-SURF数据集：Color、Depth、Infrared
    WMCA数据集：RGB、Color、Depth、Infrared、Thermal
    """
    data_cut = {}
    if config.dataset_name == 'CASIA-SURF':
        data_cut = {'train': 1, 'val': 2, 'test': 29}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name == 'WMCA':
        data_cut = {'train': 1, 'val': 10, 'test': 5}
        if config.prot == 'glasses':
            data_cut['train'] = 1

        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name in ['HQ-WMCA', 'WMCA-HQWMCA', 'HQWMCA-WMCA']:
        data_cut = {'train': 1, 'val': 1, 'test': 1}
    return data_cut

def load_list(list_path, data_cut):
    """
    获得训练集/测试集的路径和标签，并实现跳帧选择
    :param list_path: train_list/val_list/test_list的路径
    :return: list中的path和label
    """
    list = []
    f = open(list_path)
    lines = f.readlines()
    i = 0
    for line in lines:
        i = i + 1
        if i % data_cut == 0:
            line = line.strip().split(' ')
            list.append(line)
    return list


def transform_balance(train_list, data_name):
    # 实现数据集的平衡(在FAS中，一般是欺骗人脸多于真实人脸)
    print('train data balance!')
    idx = 3
    if data_name in ['WMCA', 'HQ-WMCA', 'WMCA-HQWMCA', 'HQWMCA-WMCA']:
        idx = 5
    pos_list, neg_list = [], []
    all_list = []
    for tmp in train_list:
        if int(tmp[idx]) == 1:
            label = 1
        else:
            label = 0
        if label == 1:
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)
    print(f"balance before: pos sample: {len(pos_list)}, neg sample: {len(neg_list)}")
    if int(len(neg_list)/len(pos_list)) == 0:
        neg_list = neg_list * (int(len(pos_list)/len(neg_list)))
    else:
        pos_list = pos_list * (int(len(neg_list)/len(pos_list)))
    print(f"balance after: pos sample: {len(pos_list)}, neg sample: {len(neg_list)}")
    all_list = pos_list + neg_list

    return all_list


class FAS_multi_Dataset(Dataset):
    # modality = config.image_modality, image_size = config.image_size,
    # fold_index = config.train_fold_index, balance = False, isVal = 'train'
    def __init__(self, data_root, list_path, config, balance=False, isVal='train'):
        super(FAS_multi_Dataset, self).__init__()
        self.data_root = data_root
        self.list_path = list_path
        self.data_name = config.dataset_name
        # self.modality = config.image_modality
        if getattr(config, 'strong_augment', False):
            self.augment = strong_color_augumentor
        else:
            self.augment = color_augumentor
        self.fold_index = config.train_fold_index
        self.balance = balance
        self.isVal = isVal
        self.isLocal = RESIZE_SIZE > config.image_size
        if self.isLocal:
            self.image_size = config.image_size
        else:
            self.image_size = RESIZE_SIZE
        self.channels = 0
        self.num_modalities = getattr(config, 'num_modalities', 3)
        self.map_size = int(config.image_size/16)
        self.label_weight = 0.99
        self.set_mode(config)

    def set_mode(self, config):
        data_cut = Data_Cut(config)
        if self.isVal == 'train':
            self.train_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.train_list)
            if self.balance:
                self.train_list = transform_balance(self.train_list, self.data_name)
                self.num_data = len(self.train_list)
                # print(self.train_list)
            random.shuffle(self.train_list)
            print(f'Load training set, the number of pictures is {self.num_data}')
        elif self.isVal == 'val':
            self.val_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.val_list)
            print(f'Load val set, the number of pictures is {self.num_data}')
        elif self.isVal == 'test':
            
            self.val_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.val_list)
            if self.balance:
                self.val_list = transform_balance(self.val_list, self.data_name)
                self.num_data = len(self.val_list)
                # print(self.train_list)
            print(f'Load test set, the number of pictures is {self.num_data}')

    def get_data_info(self, row):
        img_sub_path = {}
        all_image = {}
        label = None
        if self.data_name in ['WMCA', 'HQ-WMCA', 'WMCA-HQWMCA', 'HQWMCA-WMCA']:
            # RGB、Color、Depth、Infrared、Thermal
            # rgb, color, depth, ir, thermal, label = self.val_list[index]
            # img_sub_path = {'rgb': row[0], 'color': row[1], 'depth': row[2], 'ir': row[3], 'thermal': row[4]}
            img_sub_path = {'color': row[1], 'depth': row[2], 'ir': row[3], 'thermal': row[4]}
            # all_image = {'rgb': None, 'color': None, 'depth': None, 'ir': None, 'thermal': None}
            all_image = {'color': None, 'depth': None, 'ir': None, 'thermal': None}
            label = row[5]
        elif self.data_name == 'CASIA-SURF':
            # color, depth, ir, label = self.train_list[index]
            # color, depth, ir, label = self.val_list[index]
            img_sub_path = {'color': row[0], 'depth': row[1], 'ir': row[2]}
            all_image = {'color': None, 'depth': None, 'ir': None}
            label = row[3]

        for key, value in img_sub_path.items():
            all_image[key] = cv2.imread(os.path.join(self.data_root, value),1)
            # if key == 'color':
                # all_image[key] = rgb2ycbcr(all_image[key])
            # all_image[key] = cv2.resize(all_image[key], (RESIZE_SIZE, RESIZE_SIZE))
        return all_image, label

    def get_img(self, all_image, col, is_infer):
        images = None
        flag = True
        self.channels = 0
        for key, image in all_image.items():
            channel = image.shape[2]
            self.channels = self.channels + channel
            image = self.augment(image, target_shape=(self.image_size, self.image_size, channel), isLocal=self.isLocal,is_infer=is_infer)
            self.n = len(image)
            if self.isVal == 'train':
                image = cv2.resize(image, (self.image_size, self.image_size))
                image.reshape([self.image_size, self.image_size, channel])
            else:
                image.reshape([self.n, self.image_size, self.image_size, channel])
            if flag:
                flag = False
                images = image
            else:
                images = np.concatenate([images, image], axis=col)

        return images

    def __getitem__(self, index):
        if self.isVal == 'train':
            row = self.train_list[index]
            # rgb, color, depth, ir, thermal, label = self.train_list[index]
        else:
            row = self.val_list[index]
        all_image, label = self.get_data_info(row)
        
        if int(label) == 1:
            label = 1
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)
        else:
            label = 0
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)

        images = None
        if self.isVal == 'train':
            images = self.get_img(all_image, 2, False)

            if random.randint(0, 1) == 0:
                # Dropout among modalities the model actually uses
                n_mod = self.num_modalities
                random_pos = random.randint(0, n_mod - 1)
                if random.randint(0, 1) == 0:
                    images[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(n_mod):
                        if i != random_pos:
                            images[:, :, 3 * i:3 * (i + 1)] = 0

            images = np.transpose(images, (2, 0, 1))
            images = images.astype(np.float32)
            images = images.reshape([self.channels, self.image_size, self.image_size])
            images = images / 255.0

            # return torch.FloatTensor(images), torch.LongTensor(np.asarray(label).reshape([-1]))
        else:
            images = self.get_img(all_image, 3, True)
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.astype(np.float32)
            images = images.reshape([self.n, self.channels, self.image_size, self.image_size])
            images = images / 255.0

        return torch.FloatTensor(images), mask, torch.LongTensor(np.asarray(label).reshape([-1]))


    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    from augmentation import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default=-1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer_test'])
    parser.add_argument('--dataset_name', type=str, default='CASIA-SURF', help="data_name in ['WMCA', 'CASIA-SURF']")
    config = parser.parse_args()
    data_root, list_path = None, None
    data_name = 'CASIA-SURF'
    if data_name == 'CASIA-SURF':
        # OULU-NPU Dataset root
        print("========loader CASIA-SURF=========")  # 1200 videos
        data_root = 'E:/python-project/pytorch17/FAS/data/CASIA-SURF/'
        list_path = 'E:/python-project/pytorch17/FAS/data/CASIA-SURF/train_list.txt'

    image_size = 64
    isVal = 'test'
    # isVal = 'test'
    dataset = FAS_multi_Dataset(data_root, list_path, config, balance=True, isVal=isVal)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16, drop_last=True, num_workers=0)
    num = 5
    for input, truth in train_loader:
        print(input.shape, truth)
        break