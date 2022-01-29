import argparse
import glob
import os
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import ClassifyDataset
from networks import Efficient


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    ## Hyper Parameters
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size.')
    parser.add_argument('--lr', default=1e-3, type=float, help='base value of learning rate.')
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay for SGD')
    parser.add_argument('--img_size', default=[256, 256], nargs="+", type=int, help='the image size')
    ## Data Path
    parser.add_argument('--train_path', default="data/train/images", help='the train images path')
    parser.add_argument('--valid_path', default="", help='the valid images path')
    parser.add_argument('--test_path', default="data/test/images", help='the test images path')
    ## Training Resource Setting
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--multi_gpu', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--device', default=0, type=int, help='device number')
    ## Save & Load
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')
    parser.add_argument('--save_path', default='weights', help='Location to save checkpoint models')
    args = parser.parse_args()
    return args


def train(args):
    is_cuda = torch.cuda.is_available()
    device = f"cuda:{args.device}" if is_cuda else "cpu"

    # Def Loss functions
    criterion_bce = torch.nn.BCELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    # Configure dataloaders
    train_transforms = transforms.Compose([transforms.Resize((args.img_size[1], args.img_size[0])),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=(0, 360)),
                                           transforms.RandomPerspective(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           ])
    valid_transforms = transforms.Compose([transforms.Resize((args.img_size[1], args.img_size[0])),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           ])
    # imgnet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.valid_path == "":
        entire_paths = sorted(glob.glob(f"{args.train_path}/*"))
        args.valid_path = args.train_path
        random.shuffle(entire_paths)
        lab_paths_train = entire_paths[:int(len(entire_paths) * 0.9)]
        lab_paths_valid = entire_paths[int(len(entire_paths) * 0.9):]

    train_dataset = ClassifyDataset(
        args.train_path,
        transforms=train_transforms,
        mode="train",
        lab_paths=lab_paths_train,
    )
    valid_dataset = ClassifyDataset(
        args.valid_path,
        transforms=valid_transforms,
        mode="train",
        lab_paths=lab_paths_valid,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size * 2,
                              shuffle=False,
                              num_workers=args.num_workers,
                              )

    # Initialize generator and discriminator
    classifier = Efficient.Efficient(crop=len(train_dataset.dict_crops),
                                     dise=len(train_dataset.dict_dises),
                                     risk=len(train_dataset.dict_risks))
    # classifier = ResNet50(crop=len(train_dataset.dict_crops),
    #                       dise=len(train_dataset.dict_dises),
    #                       risk=len(train_dataset.dict_risks))
    if args.multi_gpu:
        classifier = torch.nn.DataParallel(classifier, device_ids=args.multi_gpu).cuda()
    elif is_cuda:
        classifier = classifier.to(device)

    if args.pretrained_model != "":
        # Load pretrained models
        pre_trained = args.pretrained_model
        classifier.load_state_dict(torch.load(pre_trained))

    total_step = len(train_loader) * args.n_epochs
    # Optimizers
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler_C = torch.optim.lr_scheduler.OneCycleLR(optimizer_C, max_lr=args.lr, total_steps=total_step,
                                                      pct_start=0.1)

    best_acc = [0, 0, 0]
    os.makedirs("result", exist_ok=True)
    result_path = f"result/{args.save_path}"
    os.makedirs(result_path, exist_ok=True)
    tb_idx = 0
    tb_dirs = "tb_logs/" + args.save_path.split("/")[-1] + f"{tb_idx}"
    while os.path.isdir(tb_dirs):
        tb_idx += 1
        tb_dirs = "tb_logs/" + args.save_path.split("/")[-1] + f"{tb_idx}"
    writer = SummaryWriter(tb_dirs)
    for epoch in range(args.n_epochs):
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Train {epoch} / {args.n_epochs}")
        for idx, batch in pbar:
            # Model inputs
            imgs_origin = batch[0].to(device)
            crops = batch[1].to(device)
            dises = batch[2].to(device)
            risks = batch[3].to(device)

            # Get Batch Size
            batch_size = len(imgs_origin)
            # ---------------------
            #  Train Classifier
            # ---------------------
            optimizer_C.zero_grad()
            output_cls = classifier(imgs_origin)
            loss_crop = criterion_ce(output_cls[0], crops)
            loss_dise = criterion_ce(output_cls[1], dises)
            loss_risk = criterion_ce(output_cls[2], risks)

            # Total loss
            loss_C = loss_crop + loss_dise + loss_risk

            # Backward & Optimize
            loss_C.backward()
            optimizer_C.step()
            scheduler_C.step()

            # --------------
            #  Log Progress
            # --------------

            step = epoch * len(train_loader) + idx
            if step % 20 == 0:
                writer.add_scalar("Train/C/total", loss_C.item(), step)
                writer.add_scalar("Train/C/crop", loss_crop.item(), step)
                writer.add_scalar("Train/C/dise", loss_dise.item(), step)
                writer.add_scalar("Train/C/risk", loss_risk.item(), step)
                writer.add_scalar("Train/C/LR", optimizer_C.param_groups[0]['lr'], step)
                writer.flush()

        # --------------
        #  Validation
        # --------------
        with torch.no_grad():
            train_acc = evaluate_classifier(classifier, train_loader, device, writer, epoch, mode="Train")
            valid_acc = evaluate_classifier(classifier, valid_loader, device, writer, epoch, mode="Valid")
            if valid_acc[0] > best_acc[0]:
                best_acc[0] = valid_acc[0]
                with open(result_path + '/best_crop.txt', 'w') as f:
                    f.write(f"{train_acc}\n{valid_acc}")
                torch.save(classifier.state_dict(), f"{result_path}/best_classifier_crop.pth")
            if valid_acc[1] > best_acc[1]:
                best_acc[1] = valid_acc[1]
                with open(result_path + '/best_dise.txt', 'w') as f:
                    f.write(f"{train_acc}\n{valid_acc}")
                torch.save(classifier.state_dict(), f"{result_path}/best_classifier_dise.pth")
            if valid_acc[2] > best_acc[2]:
                best_acc[2] = valid_acc[2]
                with open(result_path + '/best_risk.txt', 'w') as f:
                    f.write(f"{train_acc}\n{valid_acc}")
                torch.save(classifier.state_dict(), f"{result_path}/best_classifier_risk.pth")
        classifier.train()
        # --------------
        #  Save
        # --------------
        if epoch % 16 == 0:
            # Save model checkpoints
            torch.save(classifier.state_dict(), f"{result_path}/classifier_%d.pth" % epoch)


# ----------
#  Testing
# ----------

def evaluate_classifier(classifier, loader, device, writer, step, mode="Valid"):
    classifier.eval()
    # Tp, Tp_1, Tp_2 = 0, 0, 0
    # Tn_1, Tn_2 = 0, 0
    corr_crop, corr_dise, corr_risk, cnt = 0, 0, 0, 0
    for idx, batch in tqdm(enumerate(loader), desc=f"{mode} Eval", total=len(loader)):
        # load train data
        images, crops, dises, risks = batch
        images = images.to(device)
        # forward
        logits = classifier(images.detach())
        # greedy decode
        probs = [logit.softmax(dim=1).detach().cpu() for logit in logits]
        preds = [prob.argmax(dim=1).detach().numpy() for prob in probs]
        corr_crop += (preds[0] == crops.detach().cpu().numpy()).sum()
        corr_dise += (preds[1] == dises.detach().cpu().numpy()).sum()
        corr_risk += (preds[2] == risks.detach().cpu().numpy()).sum()
        cnt += len(crops)

    writer.add_scalar(f"Eval/{mode}_Crop", corr_crop / cnt, step)
    writer.add_scalar(f"Eval/{mode}_Disease", corr_dise / cnt, step)
    writer.add_scalar(f"Eval/{mode}_Risk", corr_risk / cnt, step)
    return corr_crop / cnt, corr_dise / cnt, corr_risk / cnt


def predict(args):
    ## from baseline
    crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
    crop2code = {int(k): idx for idx, k in enumerate(crop)}
    cropKey = list(crop2code.keys())
    disease = {'1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                     'b8': '다량원소결핍 (K)'},
               '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)',
                     'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
               '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                     'b8': '다량원소결핍 (K)'},
               '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                     'b8': '다량원소결핍 (K)'},
               '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                     'b8': '다량원소결핍 (K)'},
               '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}
    disease2code = {"00": 0}
    code_idx = len(disease2code)
    for dise in disease.values():
        for key, name in dise.items():
            if not key in disease2code:
                disease2code[key] = code_idx
                code_idx += 1
    diseaseKey = list(disease2code.keys())
    risk2code = {0: 0}
    code_idx = len(risk2code)
    risk = {'1': '초기', '2': '중기', '3': '말기'}
    for r in risk.keys():
        risk2code[int(r)] = code_idx
        code_idx += 1

    is_cuda = torch.cuda.is_available()
    device = f"cuda:{args.device}" if is_cuda else "cpu"

    pred_transforms = transforms.Compose([transforms.Resize((args.img_size[1], args.img_size[0])),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          ])

    entire_paths = sorted(glob.glob(f"{args.test_path}/*"))
    pred_dataset = ClassifyDataset(
        args.test_path,
        transforms=pred_transforms,
        mode="pred",
        lab_paths=entire_paths,
    )
    pred_loader = DataLoader(pred_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             )
    results_list = []
    MODEL_KINDS = ["crop", "dise", "risk"]
    for kindIdx, kind in enumerate(MODEL_KINDS):
        # classifier = ResNet50(crop=len(pred_dataset.dict_crops),
        #                        dise=len(pred_dataset.dict_dises),
        #                        risk=len(pred_dataset.dict_risks))
        classifier = Efficient.Efficient(crop=len(pred_dataset.dict_crops),
                                         dise=len(pred_dataset.dict_dises),
                                         risk=len(pred_dataset.dict_risks))
        if args.multi_gpu:
            classifier = torch.nn.DataParallel(classifier, device_ids=args.multi_gpu).cuda()
        elif is_cuda:
            classifier = classifier.to(device)
        # model_path = f"result/{args.save_path}/best_classifier_{kind}.pth"
        # classifier.load_state_dict(torch.load(model_path, map_location="cuda:0"))

        classifier.eval()
        results = []
        for idx, batch in tqdm(enumerate(pred_loader)):
            images, crops, dises, risks = batch
            images = images.to(device)
            # forward
            logits = classifier(images.detach())
            # greedy decode
            probs = [logit.softmax(dim=1).detach().cpu() for logit in logits]
            preds = [prob.argmax(dim=1).detach().numpy() for prob in probs][kindIdx]
            results.extend(preds)

        if kind == "dise":
            results = list(map(lambda diseaseIdx: diseaseKey[diseaseIdx], results))
        elif kind == "crop":
            results = list(map(lambda Idx: str(cropKey[Idx]), results))
        else:
            results = list(map(str, results))
        results_list.append(results)

    results_list = list(map(lambda idx: results_list[0][idx] + "_" + results_list[1][idx] + "_" + results_list[2][idx],
                            range(len(results_list[0]))))
    return results_list


if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()
    train(args)
    preds = predict(args)
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    submission.to_csv('jisoo_submission.csv', index=False)
