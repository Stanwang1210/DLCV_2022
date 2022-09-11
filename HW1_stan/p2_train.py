from sched import scheduler
import torch
import os
import sys
import pandas as pd
import numpy as np
from os.path import join 
import argparse
from torchvision import transforms
from dataset import P2_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from utils import fix, get_accuracy, mean_iou_score
from torch import nn
import logging
from torch.optim.lr_scheduler import StepLR
from model_p2 import Model
import torch.nn.functional as F
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("CV HW1 P2")
def main(args):


    fix(args.seed)
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model_name_or_path, pretrained=True)
    fcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model = Model(model, fcn_model)
    train_dataset = P2_Dataset(data_dir=args.data_dir, task='train', transform=transform)
    valid_dataset = P2_Dataset(data_dir=args.data_dir, task='validation', transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2]).to(device))
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        # beta=args.beta,
    )
    scheduler = StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    model = model.to(device)
    model.train()
    Train_Loss, Train_IOU, Train_total_IOU, Eval_Loss, Eval_IOU, Eval_total_IOU, = [], [], [[] for i in range(6)], [], [], [[] for i in range(6)]
    for epoch in range(args.epoch):
        p_bar = tqdm(train_loader)
        model.train()
        Loss = []
        IOU = []
        Total_iou = [[] for i in range(6)]
        for data, label, file_name in p_bar:

            output = model(data.to(device))

            loss = criterion(output, label.long().to(device))
            pred = F.softmax(output, dim = 1)
            pred = np.argmax(pred.detach().cpu().numpy(), axis = 1)
            iou, total_iou = mean_iou_score(pred, label.numpy())  
            for i in range(len(total_iou)):
                Total_iou[i].append(total_iou[i])
            loss.backward()
            Loss.append(loss.item())
            IOU.append(iou)
            optimizer.step()
            scheduler.step
            optimizer.zero_grad()
            p_bar.set_description(f'Train Loss : {loss.item():.2f}, Train IOU : {iou*100:.2f}%')
            p_bar.refresh()
        Train_Loss.append(np.mean(Loss)) 
        Train_IOU.append(np.mean(IOU)*100) 
        for i in range(len(Total_iou)):
            Train_total_IOU[i].append(np.mean(Total_iou[i])*100)
        with torch.no_grad():
            Loss = []
            IOU = []
            Total_iou = [[] for i in range(6)]
            model.eval()
            for data, label, file_name in tqdm(valid_loader):

                output = model(data.to(device))
                pred = F.softmax(output, dim = 1)
                pred = np.argmax(pred.detach().cpu().numpy(), axis = 1)
                iou, total_iou = mean_iou_score(pred, label.numpy()) 
                for i in range(len(total_iou)):
                    Total_iou[i].append(total_iou[i])
                Loss.append(loss.item())
                IOU.append(iou)
            logger.info(f'At Epoch {epoch}, Eval Loss : {np.mean(Loss):.2f}, Eval IOU : {np.mean(IOU)*100:.2f}%')
            Eval_Loss.append(np.mean(Loss)) 
            Eval_IOU.append(np.mean(IOU)*100)
            for i in range(len(Total_iou)):
                Eval_total_IOU[i].append(np.mean(Total_iou[i])*100)
    plt.plot(Train_Loss, label='train loss')
    plt.plot(Eval_Loss, label='eval loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('P2 Loss')
    plt.savefig("P2_Loss.jpg")
    plt.clf()
    plt.plot(Train_IOU, label='train iou')
    for i in range(len(Train_total_IOU)):
        plt.plot(Train_total_IOU[i], label=f'IOU_{i}')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('iou')
    plt.title('P2 Train IOU')
    plt.savefig("P2_train_IOU.jpg")
    plt.clf()
    plt.plot(Eval_IOU, label='eval iou')
    for i in range(len(Eval_total_IOU)):
        plt.plot(Eval_total_IOU[i], label=f'IOU_{i}')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('iou')
    plt.title('P2 IOU')
    plt.savefig("P2_eval_IOU.jpg")
    plt.clf()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # randomness
    parser.add_argument("--seed", type=int, default=9999)
    # model 
    parser.add_argument("--model_name_or_path", type=str, default='vgg19')

    # data
    parser.add_argument("--data_dir", type=str, default='hw1_data/p2_data')
    parser.add_argument("--batch_size", type=int, default=64)
    # hyperparams
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.5)

    args = parser.parse_args()
    main(args)