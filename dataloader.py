import os
from turtle import update
from sklearn.linear_model import lars_path
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
        

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_fold_idx(pair_data):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    idx_train, idx_valid = list(kf.split(pair_data))[0]

    return idx_train, idx_valid

def make_dataset(args, image_path, json_path):
    if args.mode == "train":
        pair_data = []
        normal_risk_list = []
        early_risk_list = []
        middle_risk_list = []
        last_risk_list = []
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
            
            if risk_label == 0:
                normal_risk_list.append(risk_label)
            elif risk_label == 1:
                early_risk_list.append(risk_label)
            elif risk_label == 2:
                middle_risk_list.append(risk_label)
            elif risk_label == 3:
                last_risk_list.append(risk_label)
            pair_data.append([img_file_path, crop_label, disease_label, risk_label])
        
        risk_set = {"normal": normal_risk_list, "early": early_risk_list, "middle": middle_risk_list, "last": last_risk_list}
        return pair_data, risk_set
    
    elif args.mode == "test":
        image_file_paths = []
        for name in os.listdir(image_path):
            name = name.split(".")[0]
            img_file_path = os.path.join(image_path, name+".jpg")
            image_file_paths.append(img_file_path)

        return image_file_paths

def make_risk_dataset(args, image_path, json_path):
    if args.mode == "train":
        normal_pair_data = []
        early_pair_data = []
        middle_pair_data = []
        last_pair_data = []
        
        for name in os.listdir(image_path):
            name = name.split(".")[0]
            img_file_path = os.path.join(image_path, name+".jpg")
            json_file_path = os.path.join(json_path, name+".json")
            with open(json_file_path, 'r') as jf:
                json_dict = json.load(jf)
                risk = json_dict["annotations"]["risk"]
            
            risk_label = risk2code[risk]
            if risk_label == 0:
                normal_pair_data.append([img_file_path, risk_label])
            elif risk_label == 1:
                early_pair_data.append([img_file_path, risk_label])
            elif risk_label == 2:
                middle_pair_data.append([img_file_path, risk_label])
            elif risk_label == 3:
                last_pair_data.append([img_file_path, risk_label])

        pair_data = normal_pair_data + early_pair_data + middle_pair_data + last_pair_data
        return pair_data
    
class DaconDataLoader(object):
    def __init__(self, args):
        self.crop2code = crop2code
        self.disease2code = disease2code
        self.risk2code = risk2code
        
        set_seeds(seed=args.num_seed)
        if args.mode == 'train':
            train_sampler = None
            
            self.training_samples = DataLoadPreprocess(args)
            idx_train_list, idx_valid_list = update_fold_idx(self.training_samples)
            
            # valid_pair_data = DataLoadPreprocess(args, mode="valid")
            
            # idx_train, idx_valid = train_test_split(list(range(len(data_samples.pair_data))),
            #                                         test_size=0.25)
            train_samples = Subset(self.training_samples, idx_train_list)
            valid_samples = Subset(self.training_samples, idx_valid_list)
            
            self.training_data = DataLoader(train_samples, args.batch_size, 
                                            shuffle=(train_sampler is None),
                                            num_workers=args.num_threads)
            
            self.validation_data = DataLoader(valid_samples, args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_threads)
        
        elif args.mode == "test":
            test_samples = DataLoadPreprocess(args)
            self.test_data = DataLoader(test_samples, args.batch_size, shuffle=False, num_workers=args.num_threads)
            
class DataLoadPreprocess(object):
    def __init__(self, args):
        self.args = args
        
        if self.args.mode == "train":
            image_path = os.path.join(args.data_path, args.mode, "images")
            json_path = os.path.join(args.data_path, args.mode, "json")
            
            # self.pair_data, self.risk_set = make_dataset(args, image_path, json_path)
            self.pair_data = make_risk_dataset(args, image_path, json_path)

        elif self.args.mode == "test":
            image_path = os.path.join(args.data_path, args.mode, "images")
            self.image_file_paths = make_dataset(args, image_path, json_path=None)
            
        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, idx):
        if self.args.mode == "train":
            # img_path, crop_label, disease_label, risk_label = self.pair_data[idx]
            img_path, label = self.pair_data[idx]
        
            image = Image.open(img_path)
            image = np.array(image, dtype=np.float32) /255.0
            image = cv2.resize(image, dsize=(384, 512))

            label = np.array(label, dtype=np.long)
            # crop_label = np.array(crop_label, dtype=np.long)
            # disease_label = np.array(disease_label, dtype=np.long)
            # risk_label = np.array(risk_label, dtype=np.long)

            image = self.train_preprocessing(image)
        
            image = transforms.ToTensor()(image)
            label = torch.tensor(label, dtype=torch.long)
            # crop_label = torch.tensor(crop_label, dtype=torch.long)
            # disease_label = torch.tensor(disease_label, dtype=torch.long)
            # risk_label = torch.tensor(risk_label, dtype=torch.long)

            image = self.normalization(image)
    
            # sample = {"image": image, "crop_lbl": crop_label, "disease_lbl": disease_label, "risk_lbl": risk_label}
            sample = {"image": image, "label": label}
            return sample

        elif self.args.mode == "test":
            img_path = self.image_file_paths[idx]
            
            image = Image.open(img_path)
            image = np.array(image, dtype=np.float32) /255.0
            image = cv2.resize(image, dsize=(384, 512))
            image = transforms.ToTensor()(image)
        
            image = self.normalization(image)
            
            return image
    
    def __len__(self):
        if self.args.mode == "train":
            return len(self.pair_data)
        
        elif self.args.mode == "test":
            return len(self.image_file_paths)
        
    def train_preprocessing(self, image):
        hflip = random.random()
        vflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
        if vflip > 0.5:
            image = (image[::-1, :, :]).copy()
        
        
        return image