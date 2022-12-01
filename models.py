import torch
import torch.nn as nn
import math


from transformers import AutoConfig, AutoModel, AutoTokenizer
import tokenization_kisti as tokenization
from preprocessing import korsci_tokenizer



class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
    
    
class LSTMBertModel(nn.Module):
    '''BERT Model Architecture.
    '''
    def __init__(self, cfg, args, num_classes=[3,9]):
        super(LSTMBertModel, self).__init__()
        
        if args.plm == 'korscibert':
            self.model_config = AutoConfig.from_pretrained('./data/bert_config_kisti.json')
            self.model = AutoModel.from_pretrained('./data/pytorch_model.bin', config=self.model_config)
        else:
            self.model_config = AutoConfig.from_pretrained(cfg.MODEL_NAME)
            self.model = AutoModel.from_pretrained(cfg.MODEL_NAME, config=self.model_config)
            
        self.num_layers = 1
        self.n_hidden = 256
        self.bidirectional = 2
        self.coarse_total = ['연구 목적', '연구 방법', '연구 결과']
        self.fine_total = ['문제 정의', '가설 설정', '기술 정의',
                               '제안 방법', '대상 데이터', '데이터처리',
                               '이론/모형', '성능/효과', '후속연구']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.label_lstm = nn.LSTM(self.model_config.hidden_size, self.n_hidden, bidirectional=True, batch_first=True)
        self.coarse_label_lstm = nn.LSTM(self.model_config.hidden_size, self.n_hidden, bidirectional=True, batch_first=True)
        self.fine_label_lstm = nn.LSTM(self.model_config.hidden_size + self.n_hidden * 2, self.n_hidden, bidirectional=True, batch_first=True)
        

        self.coarse_table = torch.tensor(self.make_label_table(cfg, args, self.coarse_total), requires_grad=True).to(self.device)
        self.fine_table = torch.tensor(self.make_label_table(cfg, args, self.fine_total), requires_grad=True).to(self.device)
        
        self.coarse_q_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.coarse_k_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.coarse_v_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)

        self.fine_q_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.fine_k_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.fine_v_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)

        # Hierarchical representation, self.model_config.hidden_size
        self.linear_lvl1 = nn.Linear(512, num_classes[0])
        self.linear_lvl2 = nn.Linear(512, num_classes[1])
        
        self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1]) 
        
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,label_0=None, label_1=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs[0]   # [16, 256, 768]

        hidden = None
        scaler = self.n_hidden ** 0.5

        """
        coarse tag predict layer
        """
        coarse_lstm_outputs, hidden = self.coarse_label_lstm(x, hidden)
        coarse_lstm_outputs = self.dropout(coarse_lstm_outputs)
        
        coarse_vector, hidden_0 = self.label_lstm(self.coarse_table.unsqueeze(0), None)
        coarse_vector = coarse_vector.view(3, -1)

        coarse_q = self.coarse_q_liner(coarse_lstm_outputs)    
        coarse_k = self.coarse_k_liner(coarse_lstm_outputs)        
        coarse_v = self.coarse_v_liner(coarse_lstm_outputs)
        coarse_attention_score = coarse_q.matmul(coarse_k.permute(0, 2, 1)) / scaler
        coarse_attention_align = self.softmax(coarse_attention_score)
        

        coarse_attention_output = coarse_attention_align.matmul(coarse_v)     
        coarse_output = coarse_attention_output.matmul(coarse_vector.permute(1,0))[:,0,:]
        
        x_2 = torch.cat([x, coarse_attention_output], dim=-1) 


        fine_lstm_outputs, _ = self.fine_label_lstm(x_2, hidden)
        fine_lstm_outputs = self.dropout(fine_lstm_outputs)  # [32, 256, 512])
        fine_vector, _ = self.label_lstm(self.fine_table.unsqueeze(0), None)
        fine_vector = fine_vector.view(9, -1)
        
        fine_q = self.fine_q_liner(fine_lstm_outputs)
        fine_k = self.fine_k_liner(fine_lstm_outputs)
        fine_v = self.fine_v_liner(fine_lstm_outputs)
        fine_attention_score = fine_q.matmul(fine_k.permute(0, 2, 1)) / scaler 
        fine_attention_align = self.softmax(fine_attention_score)
        
        fine_attention_output = fine_attention_align.matmul(fine_v)
        fine_output = fine_attention_output.matmul(fine_vector.permute(1,0))[:,0,:]

        level_1 = self.softmax_reg1(coarse_output)
        level_2 = self.softmax_reg2(torch.cat((level_1, fine_output), dim=1))

        return level_1, level_2, outputs[0], coarse_attention_output, fine_attention_output


    def make_label_table(self, cfg, args, total_label):
        max_length = 512
        if args.plm == 'korscibert':
            model_config = AutoConfig.from_pretrained('./data/bert_config_kisti.json')
            model = AutoModel.from_pretrained('./data/pytorch_model.bin', config=model_config)
            label_table = []
            for label in total_label:
                tokenizer = tokenization.FullTokenizer(vocab_file=cfg.vocab_file, do_lower_case=False, tokenizer_type="Mecab")
                
                tokens = tokenizer.tokenize(label)
                tokens = ['[CLS]'] + tokens
                tokens = tokens[:max_length-1]
                tokens = tokens + ['[SEP]']

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert len(input_ids) <= max_length

                attention_mask = [1] * len(input_ids)

                padding = [0] * (max_length - len(input_ids))

                input_ids = input_ids + padding
                attention_mask = attention_mask + padding
                outputs = model(input_ids=torch.tensor(input_ids).unsqueeze(0), attention_mask=torch.tensor(attention_mask).unsqueeze(0), token_type_ids=None)
                label_table.append(outputs[1].tolist()[0])
            
            
        else:
            model_config = AutoConfig.from_pretrained(cfg.MODEL_NAME)
            model = AutoModel.from_pretrained(cfg.MODEL_NAME, config=model_config)

            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
            label_table = []

            for label in total_label:
                toked_label = tokenizer(
                    label,
                    padding='max_length', truncation=True, max_length=max_length, return_token_type_ids=False)
                input_ids = torch.tensor(toked_label['input_ids']).unsqueeze(0)
                attention_mask = torch.tensor(toked_label['attention_mask']).unsqueeze(0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None)
                label_table.append(outputs[1].tolist()[0])
          


        return label_table
