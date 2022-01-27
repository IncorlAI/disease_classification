from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import DaconDataLoader
from sklearn.metrics import f1_score
from torchvision import transforms
from arch import Architecture_tmp
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
parser.add_argument('--batch_size',             type=int,   help='train batch size',                default=8)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=80)
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
parser.add_argument('--num_threads',    type=int,  help='number of threads to use for data loading', default=5)
parser.add_argument('--world_size',     type=int,  help='number of nodes for distributed training', default=1)
parser.add_argument('--multiprocessing_distributed',       help='Use multi-processing distributed training to launch '
                                                                'N process per node, which has N GPUs. '
                                                                'This is the fastest way to use PyTorch for either single node or '
                                                                'multi node data parallel training', default=False)

args = parser.parse_args()
a=1
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def compute_score(label, prediction):
    label = label.detach().cpu().numpy()
    prediction = prediction.argmax(1).detach().cpu().numpy()
    
    return f1_score(label, prediction, average="macro")

def validate_model(validation_data, model, criterion):
    total_valid_loss = 0
    valid_label_list = []
    valid_output_list = []
    with torch.no_grad():
        for valid_batched in tqdm(validation_data):
            valid_image = valid_batched["image"].cuda(args.gpu, non_blocking=True)
            valid_label = valid_batched["label"].cuda(args.gpu, non_blocking=True)

            prediction = model(valid_image)
            model.eval()
            valid_loss = criterion(prediction, valid_label)

            total_valid_loss += valid_loss
            
            valid_label_list += valid_label.detach().cpu().numpy().tolist()
            valid_output_list += prediction.argmax(1).detach().cpu().numpy().tolist()
            
    avg_valid_loss = total_valid_loss / len(validation_data)
    valid_f1 = f1_score(valid_label_list, valid_output_list, average="macro")

    return avg_valid_loss, valid_f1

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

    model = Architecture_tmp(model_type=args.model_name, num_classes=6)
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.to(f'cuda:{model.device_ids[0]}')
    # model.cuda()

    writer = SummaryWriter(log_dir=os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    global_step = 0
    steps_per_epoch = len(dataloader.training_data)
    epoch = global_step // steps_per_epoch
        
    while epoch < args.num_epochs:
        train_label_list = []
        train_output_list = []
        # for step, sample_batched in enumerate(zip(dataloader.training_data, dataloader.validation_data)):
        for step, train_batched in enumerate(dataloader.training_data):
            # train_batched, valid_batchead = sample_batched[0], sample_batched[1]

            optimizer.zero_grad()

            sample_image = train_batched["image"].cuda(args.gpu, non_blocking=True)
            sample_label = train_batched["label"].cuda(args.gpu, non_blocking=True)

            output = model(sample_image)
            loss = criterion(output, sample_label)
            loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = args.learning_rate * ((1 - epoch / args.num_epochs) ** 0.9)
                param_group['lr'] = current_lr

            optimizer.step()

            train_f1 = compute_score(sample_label, output)
            print_string = "[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}] | train loss: {:.5f} | train f1: {:.2f}"
            print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, loss, train_f1))

            if (global_step + 1) % args.log_freq == 0:
                writer.add_scalar('learning_rate', current_lr, global_step)
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar("train F1 score", train_f1, global_step)
                writer.add_image('input image/image/{}'.format(0), inv_normalize(sample_image[0, :]).data, global_step)

            writer.flush()
            global_step += 1
        
        valid_loss, valid_f1 = validate_model(dataloader.validation_data, model, criterion)
        writer.add_scalar("valid F1 score", valid_f1, global_step)
        print_string = "[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}] | train loss: {:.5f} | valid loss: {:.5f} | train f1: {:.3f} | valid f1: {:.3f}"
        print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, loss, valid_loss, train_f1, valid_f1))

        # print(print_string.format(epoch+1, args.num_epochs, steps_per_epoch, global_step+1, valid_loss, valid_f1))
        
        checkpoint = {'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{:07d}.pth'.format(global_step)))
        epoch += 1
        
if __name__ == "__main__":
    main()