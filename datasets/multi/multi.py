import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Multi(Dataset):
    def __init__(self, root, is_training):
        self.root = root
        self.is_training = is_training
        if self.is_training:
            test_fn = 'F:\Pycharm Projects\MVFI_ANH\data\snufilm\multitrain2.txt'
        else:
            test_fn = 'F:\Pycharm Projects\MVFI_ANH\data\snufilm\multi_test2.txt'

        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]


        if self.is_training:
            self.transforms = transforms.Compose([
                # transforms.RandomCrop(256),
                transforms.Resize(size=(256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),

            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(size=(640, 1280)),
                transforms.ToTensor()
            ])
        # print("Test dataset has %d triplets" % len(self.frame_list))

    def __getitem__(self, index):
        # Use self.test_all_images:
        imgpaths = self.frame_list[index]

        img1 = Image.open(os.path.join(self.root, imgpaths[0]))
        img2 = Image.open(os.path.join(self.root, imgpaths[1]))
        img3 = Image.open(os.path.join(self.root, imgpaths[2]))
        img4 = Image.open(os.path.join(self.root, imgpaths[3]))
        img5 = Image.open(os.path.join(self.root, imgpaths[4]))

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)
        img4 = self.transforms(img4)
        img5 = self.transforms(img5)

        imgs = [img1, img3, img5]
        gt = [img2, img4]
        # print('At point {}, imgs has {}, gt has {}. '.format(index,len(imgs), len(gt))) #Check loader
        return imgs, gt

    def __len__(self):
        return len(self.frame_list)


def check_already_extracted(vid):
    return bool(os.path.exists(vid + '/0001.png'))


def get_loader(data_root, batch_size, shuffle, num_workers):
    # data_root = 'data/SNUFILM'
    dataset = Multi(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == '__main__':
    dataset = 'D:\\KIEN\\Dataset\\Vimeo_90K\\'