import argparse
import pickle
from tqdm import tqdm
import gc
import numpy as np
import copy
import logging
from sklearn.metrics import f1_score
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import logging

from config import DefaultConfig
from models import *
from preprocessing import CustomDataset
from utils import get_optimizer, get_scheduler, seed_everything, request_logger, calculate_f1
from load_model import Model


from colorama import Fore, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
r_ = Fore.RED
sr_ = Style.RESET_ALL

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)
logging.set_verbosity_error()


def train(model, train_dataloader, valid_dataloader, cfg, args):
    logger.info("="*134)
    logger.info("{0:^7}|{1:^20}|{2:^20}|{3:^20}|{4:^20}|{5:^20}|{6:^20}".format('epoch',
                                                                                'best loss',
                                                                                'dev loss',
                                                                                'coarse micro-f1',
                                                                                'coarse macro-f1',
                                                                                'fine micro-f1',
                                                                                'fine macro-f1'))
    logger.info("="*134)
    
    # Load model...
    HLN = HierarchicalLossNetwork(hierarchical_labels=cfg.hierarchy, alpha=1, beta=0.8, device=device)
    model.parameters
    model.to(device)
    
    # Set criterion, optimizer, scheduler...
    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer, train_dataloader, args)
    
    train_total_loss = []
    train_total_coarse_f1 = []
    train_total_fine_f1 = []
    valid_total_loss = []
    valid_total_coarse_f1 = []
    valid_total_fine_f1 = []

    best_val_loss = np.inf
    best_val_f1 = -1

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")
        

        # Training loss/f1 score
        train_loss_value = 0
        train_epoch_loss = []
        train_batch_coarse_f1 = 0
        train_epoch_coarse_f1 = []
        train_batch_fine_f1 = 0
        train_epoch_fine_f1 = []

        # Validation loss/f1 score
        valid_loss_value = 0
        valid_epoch_loss = []
        valid_batch_coarse_f1 = 0
        valid_epoch_coarse_f1 = []
        valid_batch_fine_f1 = 0
        valid_epoch_fine_f1 = []

        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            item = {key: val.to(device) for key, val in items.items()}
            
            optimizer.zero_grad()
            
            preds_label_0, preds_label_1 = model(**item)
            preds = [preds_label_0, preds_label_1]
                        
            dloss = HLN.calculate_dloss(preds, [item['label_0'], item['label_1']])
            lloss = HLN.calculate_lloss(preds, [item['label_0'], item['label_1']])
            
            loss = dloss + lloss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_value += loss.item()
            train_batch_coarse_f1 += (calculate_f1(preds[0], item['label_0'])) # / cfg.TRAIN_BATCH)
            train_batch_fine_f1 += (calculate_f1(preds[1], item['label_1'])) # / cfg.TRAIN_BATCH)

            
            if (idx + 1) % cfg.TRAIN_LOG_INTERVAL == 0:
                train_epoch_coarse_f1.append(train_batch_coarse_f1/cfg.TRAIN_LOG_INTERVAL)
                train_epoch_fine_f1.append(train_batch_fine_f1/cfg.TRAIN_LOG_INTERVAL)
                train_epoch_loss.append(train_loss_value/cfg.TRAIN_LOG_INTERVAL)
                
                train_bar.set_description("Loss: {:.4f}/{:.4f}, coarse:{:.4f}/{:.4f}, fine: {:.4f}/{:.4f}".\
                    format(train_loss_value/cfg.TRAIN_LOG_INTERVAL,
                           sum(train_epoch_loss)/len(train_epoch_loss),
                           train_batch_coarse_f1/cfg.TRAIN_LOG_INTERVAL,
                           sum(train_epoch_coarse_f1)/len(train_epoch_coarse_f1),
                           train_batch_fine_f1/cfg.TRAIN_LOG_INTERVAL,
                           sum(train_epoch_fine_f1)/len(train_epoch_fine_f1)))

                train_loss_value = 0
                train_batch_coarse_f1 = 0
                train_batch_fine_f1 = 0

                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
                train_total_coarse_f1.append(sum(train_epoch_coarse_f1)/len(train_epoch_coarse_f1))
                train_total_fine_f1.append(sum(train_epoch_fine_f1)/len(train_epoch_fine_f1))

        with torch.no_grad():
            print(f"{b_}---- Validation ----{sr_}")
            model.eval()
            total_coarse_preds = []
            total_coarse_labels = []
            total_fine_preds = []
            total_fine_labels = []
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                item = {key: val.to(device) for key,val in items.items()}

                preds_label_0, preds_label_1 = model(**item)
                preds = [preds_label_0, preds_label_1]
                
                total_coarse_preds += torch.argmax(preds[0], dim=1).tolist()
                total_coarse_labels += item['label_0'].tolist()
                
                total_fine_preds += torch.argmax(preds[1], dim=1).tolist()
                total_fine_labels += item['label_1'].tolist()
                

                dloss = HLN.calculate_dloss(preds, [item['label_0'], item['label_1']])
                lloss = HLN.calculate_lloss(preds, [item['label_0'], item['label_1']])

                loss = dloss + lloss
                
                valid_loss_value += loss.item()
                valid_batch_coarse_f1 += (calculate_f1(preds[0], item['label_0'])) # / cfg.VALID_BATCH)
                valid_batch_fine_f1 += (calculate_f1(preds[1], item['label_1'])) # / cfg.VALID_BATCH)
                if (idx + 1) % cfg.VALID_LOG_INTERVAL == 0:
                    valid_epoch_coarse_f1.append(valid_batch_coarse_f1/cfg.VALID_LOG_INTERVAL)
                    valid_epoch_fine_f1.append(valid_batch_fine_f1/cfg.VALID_LOG_INTERVAL)
                    valid_epoch_loss.append(valid_loss_value/cfg.VALID_LOG_INTERVAL)
                    
                    valid_bar.set_description("Loss:{:.4f}/{:.4f}, coarse:{:.4f}/{:.4f}, fine:{:.4f}/{:.4f}".\
                        format(valid_loss_value/cfg.VALID_LOG_INTERVAL,
                               sum(valid_epoch_loss)/len(valid_epoch_loss),
                               valid_batch_coarse_f1/cfg.VALID_LOG_INTERVAL,
                               sum(valid_epoch_coarse_f1)/len(valid_epoch_coarse_f1),
                               valid_batch_fine_f1/cfg.VALID_LOG_INTERVAL,
                               sum(valid_epoch_fine_f1)/len(valid_epoch_fine_f1)))

                    valid_loss_value = 0
                    valid_batch_coarse_f1 = 0
                    valid_batch_fine_f1 = 0
                    
            
            print("{}Best Loss: {:3f} | This epoch Loss: {:3f}{}".format(g_, best_val_loss, (sum(valid_epoch_loss)/len(valid_epoch_loss)), sr_))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), cfg.DATA_PATH+args.model_name)
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))
            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
            valid_total_coarse_f1.append(sum(valid_epoch_coarse_f1)/len(valid_epoch_coarse_f1))
            valid_total_fine_f1.append(sum(valid_epoch_fine_f1)/len(valid_epoch_fine_f1))
            
            print("===== Coarse category f1-score =====")
            print(f"macro: {f1_score(total_coarse_labels, total_coarse_preds, average='macro')}")
            print(f"micro: {f1_score(total_coarse_labels, total_coarse_preds, average='micro')}")
            print(f"none: {f1_score(total_coarse_labels, total_coarse_preds, average=None)}")
            print("\n===== Fine category f1-score =====")
            print(f"macro: {f1_score(total_fine_labels, total_fine_preds, average='macro')}")
            print(f"micro: {f1_score(total_fine_labels, total_fine_preds, average='micro')}")
            print(f"none: {f1_score(total_fine_labels, total_fine_preds, average=None)}")
            
            logger.info("{0:^7}|{1:^20}|{2:^20.6f}|{3:^20.6f}|{4:^20.6f}|{5:^20.6f}|{6:^20.6f}".format(
                                                        epoch+1,
                                                        best_val_loss,
                                                        sum(valid_epoch_loss)/len(valid_epoch_loss),
                                                        f1_score(total_coarse_labels, total_coarse_preds, average='micro'),
                                                        f1_score(total_coarse_labels, total_coarse_preds, average='macro'),
                                                        f1_score(total_fine_labels, total_fine_preds, average='micro'),
                                                        f1_score(total_fine_labels, total_fine_preds, average='macro')))
            logger.info("-"*134)
        print()
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--plm', type=str, default='korscibert')
    parser.add_argument('--use_section', type=bool, default=False)
    parser.add_argument('--load_model', type=str, default='linear', help='linear, linearlstm, lstm')
    parser.add_argument('--logger_file', type=str, default='')
    parser.add_argument('--logger', type=str, default='korscibert_linear')
    parser.add_argument('--model_name', type=str, default='korscibert_linear.bin')
    
    cfg = DefaultConfig()
    args = parser.parse_args()
    logger = request_logger(f'{args.logger}', args)
    
    seed_everything(cfg.SEED)
    
    if args.plm == 'korscibert':
        if args.use_section:
            print(args.use_section)
            train_dataloader = pickle.load(open(cfg.DATA_PATH+'korscberti_section_train_dataloader.pkl', 'rb'))
            dev_dataloader = pickle.load(open(cfg.DATA_PATH+'korscberti_section_dev_dataloader.pkl', 'rb'))
        else:
            print(args.use_section)
            train_dataloader = pickle.load(open(cfg.DATA_PATH+'korscberti_train_dataloader.pkl', 'rb'))
            dev_dataloader = pickle.load(open(cfg.DATA_PATH+'korscberti_dev_dataloader.pkl', 'rb'))
            
    else:
        if args.use_section:
            print(args.use_section)
            train_dataloader = pickle.load(open(cfg.DATA_PATH+'section_train_dataloader.pkl', 'rb'))
            dev_dataloader = pickle.load(open(cfg.DATA_PATH+'section_dev_dataloader.pkl', 'rb'))
        else:
            print(args.use_section)
            train_dataloader = pickle.load(open(cfg.DATA_PATH+'train_dataloader.pkl', 'rb'))
            dev_dataloader = pickle.load(open(cfg.DATA_PATH+'dev_dataloader.pkl', 'rb'))
        
    load_model = Model(cfg, args)
    model = load_model()
    
    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}:::".format(args.load_model, args.model_name))
    logger.info("epochs:{}, optimizer:{}, plm:{}".format(args.epochs, args.optimizer, args.plm))
    logger.info("lr:{}, weight_decay:{}, eps:{}, num_warmup_steps:{}".format(args.lr, args.weight_decay, args.eps, args.num_warmup_steps))
    train(model, train_dataloader, dev_dataloader, cfg, args)
    logger.info("\n")