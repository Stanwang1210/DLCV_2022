from sched import scheduler
import torch
import os
import sys
import pandas as pd
import numpy as np
from os.path import join 
import argparse
from torchvision import transforms
from dataset import P1_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from utils import fix, get_accuracy
from torch import nn
import logging
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("CTMP main")
def main(args):


    fix(args.seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = P1_Dataset(data_dir='hw1_data/p1_data', task='train', transform=transform)
    valid_dataset = P1_Dataset(data_dir='hw1_data/p1_data', task='val', transform=transform)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model_name_or_path, pretrained=True)
    model.classifier = nn.Linear(model.classifier[0].in_features, train_dataset.class_num)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        # beta=args.beta,
    )
    scheduler = StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    model = model.to(device)
    model.train()
    for epoch in range(args.epoch):
        p_bar = tqdm(train_loader)
        model.train()
        for data, label, file_name in p_bar:

            output = model(data.to(device))
            pred = F.softmax(output, dim = -1)
            pred = np.argmax(pred.detach().cpu().numpy(), axis = -1)
            acc = get_accuracy(pred, label.numpy()) * 100
            loss = criterion(output, label.to(device))

            loss.backward()

            optimizer.step()
            scheduler.step
            optimizer.zero_grad()
            p_bar.set_description(f'Train Loss : {loss.item():.2f}, Train Acc : {acc:.2f}%')
            p_bar.refresh()
        with torch.no_grad():
            Loss = []
            Acc = []
            model.eval()
            for data, label, file_name in tqdm(valid_loader):

                output = model(data.to(device))
                pred = np.argmax(output.detach().cpu().numpy(), axis = -1)
                loss = criterion(output, label.to(device))
                acc = get_accuracy(pred, label.numpy()) * 100
                Loss.append(loss.item())
                Acc.append(acc)
            logger.info(f'At Epoch {epoch}, Eval Loss : {np.mean(Loss):.2f}, Eval Acc : {np.mean(Acc):.2f}%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # randomness
    parser.add_argument("--seed", type=int, default=9999)
    # model 
    parser.add_argument("--model_name_or_path", type=str, default='vgg19_bn')

    # data
    parser.add_argument("--data_dir", type=str, default='hw1_data/p1_data')
    parser.add_argument("--batch_size", type=int, default=64)
    # hyperparams
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.5)

    args = parser.parse_args()
    main(args)