import glob
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

## from baseline
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
crop2code = {int(k): idx for idx, k in enumerate(crop)}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
disease2code = {"00": 0}
code_idx = len(disease2code)
for dise in disease.values():
    for key, name in dise.items():
        if not key in disease2code:
            disease2code[key] = code_idx
            code_idx += 1
risk2code = {0: 0}
code_idx = len(risk2code)
risk = {'1':'초기','2':'중기','3':'말기'}
for r in risk.keys():
    risk2code[int(r)] = code_idx
    code_idx += 1
    
    
class ClassifyDataset(Dataset):
    def __init__(self, root, transforms=None, mode="train", lab_paths=None):
        self.transform = transforms
        self.mode = mode

        if lab_paths is None:
            self.folder_path = sorted(glob.glob("%s/*.txt" % root)) if mode == "val" else sorted(glob.glob("%s/*.txt" % root))
        else:
            self.folder_path = lab_paths
            
        self.dict_crops = crop2code
        self.dict_dises = disease2code
        self.dict_risks = risk2code
        
        self.samples = {}
        for fp in tqdm(self.folder_path, total=len(self.folder_path), desc="load data"):
            ## is windows
            img_name = fp.split('\\')[-1]
            ## is linux
            # img_name = fp.split('/')[-1]
            img_path = fp
            if not os.path.isfile(img_path):
                img_path = os.path.join(fp, img_name + ".jpeg")

            if self.mode != "pred":
                lab_path = img_path.split(".")[0].replace("images", "json") + ".json"
                if not os.path.isfile(lab_path):
                    lab_path = lab_path.split(".")[0] + ".jpg..json"
                if not os.path.isfile(lab_path):
                    lab_path = lab_path.split(".")[0] + ".jpeg..json"

                with open(lab_path, 'r') as f:
                    lab = json.load(f)

                task = lab['description']['task']
                code_crop = crop2code[int(lab['annotations']["crop"])]
                code_dise = disease2code[lab['annotations']["disease"]]
                code_risk = risk2code[int(lab['annotations']["risk"])]
                lab = (code_crop, code_dise, code_risk)
            else:
                lab = (-1,-1,-1)

            
            self.samples[img_name] = (img_path, lab)
        
        self.idxs = list(self.samples.keys())
        if transforms is None:
            if mode == "train":
                self.transform = transforms.Compose([
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    #                       std=[0.229, 0.224, 0.225])
                                                    ]
                                                    )
            elif mode == "val" or mode == "pred":
                self.transform = transforms.Compose([
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    #                       std=[0.229, 0.224, 0.225])
                                                    ]
                                                    )

    def __getitem__(self, idx):
        key_idx = self.idxs[idx]
        sample = self.samples[key_idx]
        
        img_path = sample[0]
        img = Image.open(img_path)
        ann = sample[1]
        code_crop, code_dise, code_risk = ann

        img = self.transform(img)
        lab_crop = torch.tensor(code_crop)
        lab_dise = torch.tensor(code_dise)
        lab_risk = torch.tensor(code_risk)
    
        return img, lab_crop, lab_dise, lab_risk

    def __len__(self):
        return len(self.idxs)