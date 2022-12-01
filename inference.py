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
torch.cuda.set_device(0)
logging.set_verbosity_error()


def inference(model, test_dataloader, args, cfg):
    """
    You can customize this code to fit your task.
    """
    logger.info("="*90)
    logger.info("{0:^10}|{1:^20}|{2:^20}|{3:^20}|{4:^20}".format('plm',
                                                                'coarse micro-f1',
                                                                'coarse macro-f1',
                                                                'fine micro-f1',
                                                                'fine macro-f1'))
    logger.info("="*90)
    
    valid_total_coarse_f1 = []
    valid_total_fine_f1 = []

    valid_batch_coarse_f1 = 0
    valid_epoch_coarse_f1 = []
    valid_batch_fine_f1 = 0
    valid_epoch_fine_f1 = []
    with torch.no_grad():
        model.parameters
        model.to(device)
        model.load_state_dict(torch.load(cfg.DATA_PATH+args.model_name, map_location=device))
        model.eval()
        total_coarse_preds = []
        total_coarse_labels = []
        total_fine_preds = []
        total_fine_labels = []
        test_bar = tqdm(test_dataloader, total=len(test_dataloader))
        for idx, items in enumerate(test_bar):
            item = {key: val.to(device) for key,val in items.items()}

            preds_label_0, preds_label_1, bert_outputs, coarse_attention_output, fine_attention_output = model(**item)
            preds = [preds_label_0, preds_label_1]

            total_coarse_preds += torch.argmax(preds[0], dim=1).tolist()
            total_coarse_labels += item['label_0'].tolist()

            total_fine_preds += torch.argmax(preds[1], dim=1).tolist()
            total_fine_labels += item['label_1'].tolist()

            valid_batch_coarse_f1 += (calculate_f1(preds[0], item['label_0'])) # / cfg.VALID_BATCH)
            valid_batch_fine_f1 += (calculate_f1(preds[1], item['label_1'])) # / cfg.VALID_BATCH)
            if (idx + 1) % cfg.VALID_LOG_INTERVAL == 0:
                valid_epoch_coarse_f1.append(valid_batch_coarse_f1/cfg.VALID_LOG_INTERVAL)
                valid_epoch_fine_f1.append(valid_batch_fine_f1/cfg.VALID_LOG_INTERVAL)

                valid_loss_value = 0
                valid_batch_coarse_f1 = 0
                valid_batch_fine_f1 = 0
        
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

        logger.info("{0:^10}|{1:^20.6f}|{2:^20.6f}|{3:^20.6f}|{4:^20.6f}".format(
                                                                        args.plm, 
                                                                        f1_score(total_coarse_labels, total_coarse_preds, average='micro'),
                                                                        f1_score(total_coarse_labels, total_coarse_preds, average='macro'),
                                                                        f1_score(total_fine_labels, total_fine_preds, average='micro'),
                                                                        f1_score(total_fine_labels, total_fine_preds, average='macro')))
        logger.info("-"*90)

    return preds

    
    
if __name__ == '__main__':
    cfg = DefaultConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_section', type=bool, default=False)
    parser.add_argument('--load_model', type=str, default='lstmbert', help='linear, linearlstm, lstm')
    parser.add_argument('--logger_file', type=str, default='./data/final_inference.log')
    parser.add_argument('--logger', type=str, default='inference')
    parser.add_argument('--plm', type=str, default='korscibert')
    parser.add_argument('--model_name', type=str, default=cfg.DATA_PATH+'final_korscibert.bin')
    parser.add_argument('--loader_name', type=str, default=cfg.DATA_PATH+'korscbert_test_dataloader.pkl')
    
    args = parser.parse_args()
    logger = request_logger(f'{args.logger}', args)
    seed_everything(cfg.SEED)

    test_dataloader = pickle.load(open(args.loader_name, 'rb'))
        
    load_model = Model(cfg, args)
    model = load_model()
    
    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}, {}:::".format(args.load_model, args.model_name))
    preds = inference(model, test_dataloader, args, cfg)

    logger.info("\n")