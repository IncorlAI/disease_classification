import os
import json
from shutil import copyfile
import glob

from PIL import Image
from tqdm import tqdm


SRC_DATA_ROOT = "./Externaldata/"
TAR_DATA_ROOT = "./data/Preprocessed/"
TRAIN_FOLDER = "train_ex/"


crop_map = {4:'1',6:'4',1:'5',12:'6'}
disease_map = {'1':{7:'a1',8:'a2'},
           '4': {11: 'a3', 12: 'a4'},
           '5': {1: 'a7', 2: 'a8'},
           '6':{20:'a12'}}

def makePath(path_label):
    isExist = os.path.exists(path_label)
    if not isExist:
        os.makedirs(path_label)


def saveResizedImages(src_path, tar_path):
    imgfile_names = os.listdir(src_path)
    for sample in tqdm(imgfile_names):
        fname = sample.split(".")[0]
        img = Image.open(src_path + sample)
        RESIZE_HEIGHT = 512
        size_ratio = img.height / RESIZE_HEIGHT
        resize_width = int(img.width / size_ratio)
        img = img.resize((resize_width,RESIZE_HEIGHT))
        img.save(tar_path + fname + ".jpg")


def saveRemappedJsons(src_path, tar_path):
    lbl_file_names = os.listdir(src_path)

    for sample in tqdm(lbl_file_names):
        fname = sample.split(".")[0]
        with open(src_path + sample) as file:
            json_obj = json.load(file)

        json_obj['annotations']['crop'] = crop_map[json_obj['annotations']['crop']]
        json_obj['annotations']['disease'] = disease_map[json_obj['annotations']['crop']][
            json_obj['annotations']['disease']]
        with open(tar_path + fname + '.json', 'w') as file:
            json.dump(json_obj, file)


if __name__ == '__main__':
    src_path = SRC_DATA_ROOT + TRAIN_FOLDER
    src_list = os.listdir(src_path + 'img/')

    for idx, fruit_dir in enumerate(src_list):

        img_src_path = src_path + 'img/' + fruit_dir + '/'
        lbl_src_path = src_path + 'label/[라벨]' + fruit_dir + '/'
        img_tar_path = TAR_DATA_ROOT + TRAIN_FOLDER + "img/"
        lbl_tar_path = TAR_DATA_ROOT + TRAIN_FOLDER + "label/"
        saveResizedImages(img_src_path,img_tar_path)
        saveRemappedJsons(lbl_src_path, lbl_tar_path)
