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
from urllib3 import disable_warnings


disease_dict = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},'5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}

crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}

crop2code = {int(k): idx for idx, k in enumerate(crop)}

disease2code = {"00": 0}
code_idx = len(disease2code)
for dise in disease_dict.values():
    for key, name in dise.items():
        if not key in disease2code:
            disease2code[key] = code_idx
            code_idx += 1

risk2code = {0: 0}
code_idx = len(risk2code)
risk_dict = {'1':'초기','2':'중기','3':'말기'}
for r in risk_dict.keys():
    risk2code[int(r)] = code_idx
    code_idx += 1
        

def update_fold_idx(training_samples):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    idx_train, idx_valid = list(kf.split(training_samples.pair_data))[0]

    return idx_train, idx_valid

def make_dataset(image_path, json_path):
    pair_data = []
    for name in os.listdir(image_path):
        name = name.split(".")[0]
        img_file_path = os.path.join(image_path, name+".jpg")
        json_file_path = os.path.join(json_path, name+".json")
        with open(json_file_path, 'r') as jf:
            json_dict = json.load(jf)
            crop = int(json_dict["annotations"]["crop"])
            disease = json_dict["annotations"]["disease"]
            risk = json_dict["annotations"]["risk"]
        
        crop_label = crop2code[crop]
        disease_label = disease2code[disease]
        risk_label = risk2code[risk]
        
        pair_data.append([img_file_path, crop_label, disease_label, risk_label])
    return pair_data

class DaconDataLoader(object):
    def __init__(self, args):
        if args.mode == 'train':
            train_sampler = None
            self.crop2code = crop2code
            self.disease2code = disease2code
            self.risk2code = risk2code
            
            self.training_samples = DataLoadPreprocess(args)
            idx_train_list, idx_valid_list = update_fold_idx(self.training_samples)

            # train_samples = Subset(self.training_samples, ivalidx_train_list)
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
        image_path = os.path.join(args.data_path, args.mode, "images")
        json_path = os.path.join(args.data_path, args.mode, "json")
        
        self.pair_data = make_dataset(image_path, json_path)

    def __getitem__(self, idx):
        # img_path, label = self.pair_data[idx]
        img_path, crop_label, disease_label, risk_label = self.pair_data[idx]
        image = Image.open(img_path)
        
        image = np.array(image, dtype=np.float32) /255.0
        crop_label = np.array(crop_label, dtype=np.long)
        disease_label = np.array(disease_label, dtype=np.long)
        risk_label = np.array(risk_label, dtype=np.long)
        
        label_set = {"crop": crop_label, "disease": disease_label, "risk": risk_label}
        
        if self.args.mode == "train":
            image, crop_label, disease_label, risk_label = self.train_preprocessing(image, label_set)
        
        # sample = {"image": image, "label": label}
        sample = {"image": image, "crop_lb": crop_label, "disease_lb": disease_label, "risk_lb": risk_label}
        return sample
    
    def __len__(self):
        return len(self.pair_data)

    def train_preprocessing(self, image, label_set):
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

        crop_label = torch.tensor(label_set["crop"], dtype=torch.long)
        disease_label = torch.tensor(label_set["disease"], dtype=torch.long)
        risk_label = torch.tensor(label_set["risk"], dtype=torch.long)
        image = normalization(image)
        
        return image, crop_label, disease_label, risk_label