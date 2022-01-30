import enum
from numpy import average
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import DaconDataLoader
from sklearn.metrics import f1_score
from torchvision import transforms
from arch import Architecture_tmp
from networks.resnet import ResNet50
import argparse
import os
import warnings
import torch
import torch.nn as nn
import numpy as np


parser = argparse.ArgumentParser(description="DACON")

parser.add_argument('--mode',                   type=str,   help='training and validation mode',    default='train')
parser.add_argument('--model_name',             type=str,   help='model name to be trained',        default='resnet')
parser.add_argument("--data_path",              type=str,   help="image data path",                 default=os.path.join(os.getcwd(), "data"))

# Training
parser.add_argument('--num_seed',               type=int,   help='random seed number',              default=1)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                default=16)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=50)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate',           default=1e-4)
parser.add_argument('--weight_decay',           type=float, help='weight decay factor for optimization',                                default=1e-5)
parser.add_argument('--retrain',                type=bool,  help='If used with checkpoint_path, will restart training from step zero',  default=False)
parser.add_argument('--do_eval',                type=bool,  help='Mod to evaluating the training model',                                default=False)

# Preprocessing
parser.add_argument('--random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation',   default=False)
parser.add_argument('--degree',                 type=float, help='random rotation maximum degree',                          default=2.5)

# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',               default='')
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',         default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_freq',               type=int,   help='Logging frequency in global steps',                   default=100)
parser.add_argument('--save_freq',              type=int,   help='Checkpoint saving frequency in global steps',         default=500)

# Multi-gpu training
parser.add_argument('--gpu',            type=int,  help='GPU id to use', default=0)
parser.add_argument('--rank',           type=int,  help='node rank(tensor dimension)for distributed training', default=0)
parser.add_argument('--dist_url',       type=str,  help='url used to set up distributed training', default='file:///c:/MultiGPU.txt')
parser.add_argument('--dist_backend',   type=str,  help='distributed backend', default='gloo')
parser.add_argument('--num_threads',    type=int,  help='number of threads to use for data loading', default=4)
parser.add_argument('--world_size',     type=int,  help='number of nodes for distributed training', default=1)
parser.add_argument('--multiprocessing_distributed',       help='Use multi-processing distributed training to launch '
                                                                'N process per node, which has N GPUs. '
                                                                'This is the fastest way to use PyTorch for either single node or '
                                                                'multi node data parallel training', default=False)

args = parser.parse_args()

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def compute_score(label, prediction):
    label = label.detach().cpu().numpy()
    prediction = prediction.argmax(1).detach().cpu().numpy()
    
    return f1_score(label, prediction, average="macro")

def validate_model(validation_data, model, criterion, writer, global_step):
    
    total_valid_loss = 0
    with torch.no_grad():
        model.eval()
        for step, valid_batched in enumerate(tqdm(validation_data)):
            valid_image = valid_batched["image"].cuda(args.gpu, non_blocking=True)
            valid_crop_label = valid_batched["crop_lbl"].cuda(args.gpu, non_blocking=True)
            valid_disease_label = valid_batched["disease_lbl"].cuda(args.gpu, non_blocking=True)
            valid_risk_label = valid_batched["risk_lbl"].cuda(args.gpu, non_blocking=True)
            
            logits = model(valid_image)
            
            valid_crop_label_loss = criterion(logits[0], valid_crop_label)
            valid_disease_loss = criterion(logits[1], valid_disease_label)
            valid_risk_loss = criterion(logits[2], valid_risk_label)

            total_valid_loss = valid_crop_label_loss + valid_disease_loss + valid_risk_loss
            
            crop_f1 = compute_score(label=valid_crop_label, prediction=logits[0])
            disease_f1 = compute_score(label=valid_disease_label, prediction=logits[1])
            risk_f1 = compute_score(label=valid_risk_label, prediction=logits[2])
            
            # writer.add_scalar("validation/loss", total_valid_loss, global_step)
            # writer.add_scalar("validation/crop_f1", )
            # for num in range(valid_image.shape[0]):
            #   writer.add_image("validation/image/{}".format(num), valid_image[num, :], global_step)
    return total_valid_loss, crop_f1, disease_f1, risk_f1

def main():
    # make direcotry
    command = "mkdir " + os.path.join(args.log_directory, args.model_name, "model")
    os.system(command)

    warnings.filterwarnings('ignore')
    # gpu setting
    torch.cuda.empty_cache()
    main_worker()

def main_worker():
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    dataloader = DaconDataLoader(args)

    # model = Architecture_tmp(model_type=args.model_name, num_classes=6)
    model = ResNet50(crop=len(dataloader.crop2code),
                     dise=len(dataloader.disease2code),
                     risk=len(dataloader.risk2code))
    
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.to(f'cuda:{model.device_ids[0]}')
    # model.cuda()

    writer = SummaryWriter(log_dir=os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    global_step = 0
    steps_per_epoch = len(dataloader.training_data)
    epoch = global_step // steps_per_epoch
        
    while epoch < args.num_epochs:
        for step, train_batched in enumerate(dataloader.training_data):
            # if step == 10: break
            optimizer.zero_grad()

            sample_image = train_batched["image"].cuda(args.gpu, non_blocking=True)
            sample_crop = train_batched["crop_lbl"].cuda(args.gpu, non_blocking=True)
            sample_disease = train_batched["disease_lbl"].cuda(args.gpu, non_blocking=True)
            sample_risk = train_batched["risk_lbl"].cuda(args.gpu, non_blocking=True)

            output = model(sample_image)
            loss_crop = criterion(output[0], sample_crop)
            loss_disease = criterion(output[1], sample_disease)
            loss_risk = criterion(output[2], sample_risk)
            
            total_loss = loss_crop + loss_disease + loss_risk
            total_loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = args.learning_rate * ((1 - epoch / args.num_epochs) ** 0.9)
                param_group['lr'] = current_lr

            optimizer.step()

            print_string = "[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}] | train loss: {:.5f}"
            print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, total_loss))

            if (global_step + 1) % args.log_freq == 0:
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                writer.add_scalar('train/loss', total_loss, global_step)
                for num in range(sample_image.shape[0]):
                    writer.add_image('train/image/{}'.format(num), inv_normalize(sample_image[num, :]).data, global_step)

            writer.flush()
            global_step += 1
        
        # total_valid_loss, crop_f1, disease_f1, risk_f1 = validate_model(dataloader.validation_data, model, criterion, writer, global_step)
        total_valid_loss = 0
        total_val_crop_f1 = 0
        total_val_disease_f1 = 0
        total_val_risk_f1 = 0
        with torch.no_grad():
            model.eval()
            for step, valid_batched in enumerate(tqdm(dataloader.validation_data)):
                valid_image = valid_batched["image"].cuda(args.gpu, non_blocking=True)
                valid_crop_label = valid_batched["crop_lbl"].cuda(args.gpu, non_blocking=True)
                valid_disease_label = valid_batched["disease_lbl"].cuda(args.gpu, non_blocking=True)
                valid_risk_label = valid_batched["risk_lbl"].cuda(args.gpu, non_blocking=True)
                
                logits = model(valid_image)
                
                valid_crop_label_loss = criterion(logits[0], valid_crop_label)
                valid_disease_loss = criterion(logits[1], valid_disease_label)
                valid_risk_loss = criterion(logits[2], valid_risk_label)

                total_valid_loss = valid_crop_label_loss + valid_disease_loss + valid_risk_loss
                
                crop_f1 = compute_score(label=valid_crop_label, prediction=logits[0])
                disease_f1 = compute_score(label=valid_disease_label, prediction=logits[1])
                risk_f1 = compute_score(label=valid_risk_label, prediction=logits[2])
                total_val_crop_f1 += crop_f1
                total_val_disease_f1 += disease_f1
                total_val_risk_f1 += risk_f1
                
                # for num in range(valid_image.shape[0]):
                #     writer.add_image("validation/image/{}".format(num), inv_normalize(valid_image[num, :]), global_step)
            
            total_val_crop_f1 = total_val_crop_f1 / len(dataloader.validation_data)
            total_val_disease_f1 = total_val_disease_f1 / len(dataloader.validation_data)
            total_val_risk_f1 = total_val_risk_f1 / len(dataloader.validation_data)
            
            writer.add_scalar("validation/loss", total_valid_loss, global_step)
            writer.add_scalar("validation/crop_f1", total_val_crop_f1, global_step)
            writer.add_scalar("validation/disease_f1", total_val_disease_f1, global_step)
            writer.add_scalar("validation/risk_f1", total_val_risk_f1, global_step)
        
        print_string = "[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}] | train loss: {:.5f} | valid loss: {:.5f} | crop f1: {:.3f} | disease f1: {:.3f} | risk f1: {:.3f}"
        # print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, total_loss, total_valid_loss, crop_f1, disease_f1, risk_f1))
        print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, total_loss, total_valid_loss, total_val_crop_f1, total_val_disease_f1, total_val_risk_f1))

        checkpoint = {'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{:07d}.pth'.format(global_step)))
        
        model.train()
        epoch += 1
        
if __name__ == "__main__":
    main()