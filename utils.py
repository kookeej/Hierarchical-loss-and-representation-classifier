import numpy as np
import logging

import torch
import torch.optim as optim
import transformers

from torchmetrics import F1Score

# set random seed 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
      
# optimizer
def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
     
    return optimizer

# scheduler
def get_scheduler(optimizer, train_dataloader, args, name='linear'):
    if name == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    elif name == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    return scheduler



def calculate_f1(preds, target):
    '''Calculates the accuracy of the prediction.
    '''
    num_classes = preds.size()[1]
    predicted = torch.argmax(preds, dim=1)

    f1_score = F1Score(num_classes)
    score = f1_score(predicted.cpu(), target.cpu())

    return score


def request_logger(logger_name, args):
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) > 0:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(args.logger_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger