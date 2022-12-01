import pickle
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoTokenizer
import tokenization_kisti as tokenization

from torch.utils.data import DataLoader, Dataset

from utils import *
from config import DefaultConfig


# Tokenizer
def korsci_tokenizer(sentences, max_length, return_token_type_ids, vocab_file):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False, tokenizer_type="Mecab")
    all_tokens = []
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    inputs_dict = {}
    
    for i in tqdm(range(len(sentences))):
        tokens = tokenizer.tokenize(sentences[i])
        tokens = ['[CLS]'] + tokens
        tokens = tokens[:max_length-1]
        tokens = tokens + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        input_ids = input_ids + padding
        attention_mask = attention_mask + padding
        token_type_ids = token_type_ids + padding

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
    
    if return_token_type_ids:
        inputs_dict['input_ids'] = torch.tensor(all_input_ids)
        inputs_dict['attention_mask'] = torch.tensor(all_attention_mask)
        inputs_dict['token_type_ids'] = torch.tensor(all_token_type_ids)
    else:
        inputs_dict['input_ids'] = torch.tensor(all_input_ids)
        inputs_dict['attention_mask'] = torch.tensor(all_attention_mask)   
    
    return inputs_dict
    

# Dataset
class CustomDataset(Dataset):
    """
    This is the code that creates the dataset format to put in the dataloader.
    """
    def __init__(self, tokenized_dataset, labels, length):
        self.tokenized_dataset = tokenized_dataset
        self.length = length
        self.labels = labels
        self.coarse_labels = ['연구 목적', '연구 방법', '연구 결과']
        self.fine_labels = ['문제 정의', '가설 설정', '기술 정의',
                   '제안 방법', '대상 데이터', '데이터처리',
                   '이론/모형', '성능/효과', '후속연구']

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item['label_0'] = torch.tensor(self.coarse_labels.index(self.labels[0][idx]))
        item['label_1'] = torch.tensor(self.fine_labels.index(self.labels[1][idx]))
        return item

    def __len__(self):
        return self.length
    

    
def pro_dataset(dataset, batch_size, vocab_file, args, use_section=False, model='korscibert', mode='train'):
    """
    This is the code for tokenizing, creating a custom dataset, and creating a dataloader.
    """
    sentences= dataset['sentence'].tolist()
    label_0 = dataset['tag_0'].tolist()
    label_1 = dataset['tag_1'].tolist()
    length = len(sentences)
    if use_section:
        sections = dataset['section'].tolist()
    
    print("Tokenizing...")
    
    if model == 'korscibert':
        if use_section:
#             new_sentences = []
#             for i in range(len(sentences)):
#                 s = sections[i] + '[SEP]' + sentences[i]
#                 new_sentences.append(s)
            tokenized = korsci_tokenizer(
                sections,
                sentences,
                max_length=args.max_length,
                return_token_type_ids=False,
                vocab_file=vocab_file
            )
        else:
            tokenized = korsci_tokenizer(
                sentences,
                max_length=args.max_length,
                return_token_type_ids=False,
                vocab_file=vocab_file
            )
       
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        if use_section:
            tokenized = tokenizer(
                sections,
                sentences,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=args.max_length,
                return_token_type_ids=False
            )
        else:
            tokenized = tokenizer(
                sentences,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=args.max_length,
                return_token_type_ids=False
            )   
    
    custom_dataset = CustomDataset(tokenized, (label_0, label_1), length)
    
    if mode == "train" or mode == "dev":
        OPT = True
    else:
        OPT = False
    
    print("Creating Dataloader...")
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    print("Finish!")
    
    return dataloader
    

    
if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plm', type=str, default='korscibert')
    parser.add_argument('--max_length', type=int, default=256)
    
    cfg = DefaultConfig()
    args = parser.parse_args()
    
    
    hierarchy = {
        '연구 목적': ['문제 정의', '가설 설정', '기술 정의'],
        '연구 방법': ['제안 방법', '대상 데이터', '데이터처리', '이론/모형'],
        '연구 결과': ['성능/효과', '후속연구']
    }

    mapping = {
        '문제 정의': '연구 목적',
        '가설 설정': '연구 목적',
        '기술 정의': '연구 목적',
        '제안 방법': '연구 방법',
        '대상 데이터': '연구 방법',
        '데이터처리': '연구 방법',
        '이론/모형': '연구 방법',
        '성능/효과': '연구 결과',
        '후속연구': '연구 결과'
    }

    dataset = pd.read_json(cfg.FILE_PATH, lines=False)

    dataset['tag_0'] = dataset['tag'].tolist()
    dataset['tag_0'] = dataset['tag_0'].map(mapping)
    dataset = dataset.rename(columns={'tag': 'tag_1'})
    dataset = dataset[['doc_id', 'tag_0', 'tag_1', 'sentence', 'keysentence']]
    
    section_dataset = pickle.load(open('./data/section_final.pkl', 'rb'))
    section_dataset = section_dataset.replace({'분석방법': '데이터처리'})
    section_dataset = section_dataset.rename(columns={'root_tag': 'tag_0', 'tag': 'tag_1'})
    
    isection_dataset = pickle.load(open('./data/data_sidx_cleaned.pkl', 'rb'))
    isection_dataset = isection_dataset.replace({'분석방법': '데이터처리'})
    isection_dataset = isection_dataset.rename(columns={'root_tag': 'tag_0', 'tag': 'tag_1'})
    
    train_dataset, dev_test_dataset = train_test_split(dataset, test_size=0.1, random_state=cfg.SEED)
    dev_dataset, test_dataset = train_test_split(dev_test_dataset, test_size=0.5, random_state=cfg.SEED)
    del dev_test_dataset
    print(train_dataset['tag_1'].value_counts())
    print(dev_dataset['tag_1'].value_counts())
    print(test_dataset['tag_1'].value_counts())
    
    strain_dataset, sdev_test_dataset = train_test_split(section_dataset, test_size=0.1, random_state=cfg.SEED)
    sdev_dataset, stest_dataset = train_test_split(sdev_test_dataset, test_size=0.5, random_state=cfg.SEED)
    del sdev_test_dataset
    print(strain_dataset['tag_1'].value_counts())
    print(sdev_dataset['tag_1'].value_counts())
    print(stest_dataset['tag_1'].value_counts())
    
    istrain_dataset, isdev_test_dataset = train_test_split(isection_dataset, test_size=0.1, random_state=cfg.SEED)
    isdev_dataset, istest_dataset = train_test_split(isdev_test_dataset, test_size=0.5, random_state=cfg.SEED)
    del isdev_test_dataset
    print(istrain_dataset['tag_1'].value_counts())
    print(isdev_dataset['tag_1'].value_counts())
    print(istest_dataset['tag_1'].value_counts())
    
    if args.plm == 'korscibert':
        train_dataloader = pro_dataset(train_dataset, batch_size=cfg.TRAIN_BATCH, use_section=False, model='korscibert',vocab_file=cfg.vocab_file, mode='train', args=args)
        dev_dataloader = pro_dataset(dev_dataset, batch_size=cfg.DEV_BATCH, use_section=False, model='korscibert', vocab_file=cfg.vocab_file, mode='dev', args=args)
        test_dataloader = pro_dataset(test_dataset, batch_size=cfg.TEST_BATCH, use_section=False, model='korscibert',vocab_file=cfg.vocab_file, mode='test', args=args)
        
        print("Save DataLoader...")
        pickle.dump(train_dataloader, open(cfg.DATA_PATH+'korscbert_train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dev_dataloader, open(cfg.DATA_PATH+'korscbert_dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataloader, open(cfg.DATA_PATH+'korscbert_test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        strain_dataloader = pro_dataset(strain_dataset, batch_size=cfg.TRAIN_BATCH, use_section=True, model='korscibert',vocab_file=cfg.vocab_file, mode='train', args=args)
        sdev_dataloader = pro_dataset(sdev_dataset, batch_size=cfg.DEV_BATCH, use_section=True, model='korscibert', vocab_file=cfg.vocab_file, mode='dev', args=args)
        stest_dataloader = pro_dataset(stest_dataset, batch_size=cfg.TEST_BATCH, use_section=True, model='korscibert',vocab_file=cfg.vocab_file, mode='test', args=args)
        
        print("Save DataLoader...")
        pickle.dump(strain_dataloader, open(cfg.DATA_PATH+'korscbert_section_train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sdev_dataloader, open(cfg.DATA_PATH+'korscbert_section_dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(stest_dataloader, open(cfg.DATA_PATH+'korscbert_section_test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        istrain_dataloader = pro_dataset(istrain_dataset, batch_size=cfg.TRAIN_BATCH, use_section=True, model='korscibert',vocab_file=cfg.vocab_file, mode='train', args=args)
        isdev_dataloader = pro_dataset(isdev_dataset, batch_size=cfg.DEV_BATCH, use_section=True, model='korscibert', vocab_file=cfg.vocab_file, mode='dev', args=args)
        istest_dataloader = pro_dataset(istest_dataset, batch_size=cfg.TEST_BATCH, use_section=True, model='korscibert',vocab_file=cfg.vocab_file, mode='test', args=args)
        
        print("Save DataLoader...")
        pickle.dump(istrain_dataloader, open(cfg.DATA_PATH+'korscbert_isection_train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(isdev_dataloader, open(cfg.DATA_PATH+'korscbert_isection_dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(istest_dataloader, open(cfg.DATA_PATH+'korscbert_isection_test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
    else:
        train_dataloader = pro_dataset(train_dataset, batch_size=cfg.TRAIN_BATCH, use_section=False, model='roberta',vocab_file=cfg.vocab_file, mode='train', args=args)
        dev_dataloader = pro_dataset(dev_dataset, batch_size=cfg.DEV_BATCH, use_section=False, model='roberta', vocab_file=cfg.vocab_file, mode='dev', args=args)
        test_dataloader = pro_dataset(test_dataset, batch_size=cfg.TEST_BATCH, use_section=False, model='roberta',vocab_file=cfg.vocab_file, mode='test', args=args)

        print("Save DataLoader...")
        pickle.dump(train_dataloader, open(cfg.DATA_PATH+'train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dev_dataloader, open(cfg.DATA_PATH+'dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataloader, open(cfg.DATA_PATH+'test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
        strain_dataloader = pro_dataset(strain_dataset, batch_size=cfg.TRAIN_BATCH, use_section=True, model='roberta',vocab_file=cfg.vocab_file, mode='train', args=args)
        sdev_dataloader = pro_dataset(sdev_dataset, batch_size=cfg.DEV_BATCH, use_section=True, model='roberta', vocab_file=cfg.vocab_file, mode='dev', args=args)
        stest_dataloader = pro_dataset(stest_dataset, batch_size=cfg.TEST_BATCH, use_section=True, model='roberta',vocab_file=cfg.vocab_file, mode='test', args=args)
        
        print("Save DataLoader...")
        pickle.dump(strain_dataloader, open(cfg.DATA_PATH+'section_train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sdev_dataloader, open(cfg.DATA_PATH+'section_dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(stest_dataloader, open(cfg.DATA_PATH+'section_test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
        
        istrain_dataloader = pro_dataset(istrain_dataset, batch_size=cfg.TRAIN_BATCH, use_section=True, model='roberta',vocab_file=cfg.vocab_file, mode='train', args=args)
        isdev_dataloader = pro_dataset(isdev_dataset, batch_size=cfg.DEV_BATCH, use_section=True, model='roberta', vocab_file=cfg.vocab_file, mode='dev', args=args)
        istest_dataloader = pro_dataset(istest_dataset, batch_size=cfg.TEST_BATCH, use_section=True, model='roberta',vocab_file=cfg.vocab_file, mode='test', args=args)
        
        print("Save DataLoader...")
        pickle.dump(istrain_dataloader, open(cfg.DATA_PATH+'isection_train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(isdev_dataloader, open(cfg.DATA_PATH+'isection_dev_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(istest_dataloader, open(cfg.DATA_PATH+'isection_test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    