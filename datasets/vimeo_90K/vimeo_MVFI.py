import os.path
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

class VimeoSepTuplet_MVFI(Dataset):
    def __init__(self, root, is_training):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        Outputs
            frames: list of first 3 frames
            gt : 2 grouth truth frames
        """
        self.root = root
        self.image_root = os.path.join(self.root, 'sequences')
        self.is_training = is_training

        train_fn = os.path.join(self.root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.is_training:
            self.transforms = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.is_training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 6)]
        # Load images
        list_images = [Image.open(img) for img in imgpaths]

        # Data augmentation
        if self.is_training:
            seed = random.randint(0, 2 ** 32)
            images_ = []
            for img_ in list_images:
                random.seed(seed)
                images_.append(self.transforms(img_))
            list_images = images_

            # Random Temporal Flip Order
            if random.random() >= 0.5:
                list_images = list_images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            list_images = [self.transforms(img_) for img_ in list_images]

        # Select relevant inputs: 1, 3, 5
        inputs = [0, 2, 4]

        # Check for list index out of range
        images = [list_images[i] for i in inputs]

        # Select ground truth: 2, 4
        ground_truth_indices = [1, 3]
        gt = [list_images[i] for i in ground_truth_indices]

        # return images
        return images, gt

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def get_loader(mode, data_root, batch_size, shuffle, num_workers):
    if mode == 'train':
        is_training =True
    else:
        is_training = False
    dataset = VimeoSepTuplet_MVFI(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

if __name__ == '__main__':
    dataset = 'D:\\KIEN\\Dataset\\Vimeo_90K\\'