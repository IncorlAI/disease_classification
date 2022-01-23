import os
import torch
import numpy as np
import cv2
import random
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold

def update_fold_idx(training_samples):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    idx_train, idx_valid = list(kf.split(training_samples.pair_data))[0]

    return idx_train, idx_valid

def make_dataset(image_path, json_path):
    pair_data = []
    crop_label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    for name in os.listdir(image_path):
        name = name.split(".")[0]
        img_file_path = os.path.join(image_path, name+".jpg")
        json_file_path = os.path.join(json_path, name+".json")
        with open(json_file_path, 'r') as jf:
            json_dict = json.load(jf)
            crop = json_dict["annotations"]["crop"]
        pair_data.append([img_file_path, crop_label_dict[crop]])
    return pair_data

class DaconDataLoader(object):
    def __init__(self, args):
        if args.mode == 'train':
            train_sampler = None
            
            self.training_samples = DataLoadPreprocess(args)
            idx_train_list, idx_valid_list = update_fold_idx(self.training_samples)

            # train_samples = Subset(self.training_samples, idx_train_list)
            # valid_samples = Subset(self.training_samples, idx_valid_list)
            train_sampler, valid_sampler = SubsetRandomSampler(idx_train_list), SubsetRandomSampler(idx_valid_list)
            self.training_data = DataLoader(self.training_samples, args.batch_size, 
                                            shuffle=(train_sampler is None),
                                            num_workers=args.num_threads,
                                            pin_memory= True,
                                            sampler=train_sampler)
            
            self.validation_data = DataLoader(self.training_samples, args.batch_size, 
                                              shuffle=False,
                                              num_workers=args.num_threads,
                                              pin_memory=True,
                                              sampler=valid_sampler)
            
class DataLoadPreprocess(object):
    def __init__(self, args):
        self.args = args
        image_path = os.path.join(args.data_path, "images")
        json_path = os.path.join(args.data_path, "json")
        
        self.pair_data = make_dataset(image_path, json_path)

    def __getitem__(self, idx):
        img_path, label = self.pair_data[idx]
        image = Image.open(img_path)
        
        image = np.array(image, dtype=np.float32) /255.0
        label = np.array(label, dtype=np.long)
        
        if self.args.mode == "train":
            image, label = self.train_preprocessing(image, label)
        
        sample = {"image": image, "label": label}
        return sample
    
    def __len__(self):
        return len(self.pair_data)

    def train_preprocessing(self, image, label):
        image = cv2.resize(image, dsize=(384, 512))
        hflip = random.random()
        vflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
        if vflip > 0.5:
            image = (image[::-1, :, :]).copy()
        
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        image = transforms.ToTensor()(image)
        # image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        image = normalization(image)
        
        return image, label