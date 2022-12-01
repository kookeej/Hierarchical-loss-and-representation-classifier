import os

if __name__ == '__main__':
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 2e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm korscibert \
                               --load_model linearlstm \
                               --logger korscibert_linear \
                               --logger_file ./data/experiments_false.log \
                               --model_name korscibert_linearlstm.bin")
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 2e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm korscibert \
                               --load_model lstmcopy1 \
                               --logger_file ./data/experiments_false.log \
                               --logger korscibert_lstmcp1 \
                               --model_name korscibert_lstmcp1.bin")

    
    
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 2e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm roberta \
                               --load_model linearlstm \
                               --logger roberta_linear \
                               --logger_file ./data/experiments_false.log \
                               --model_name roberta_linearlstm.bin")
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 2e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm roberta \
                               --load_model lstmcopy1 \
                               --logger_file ./data/experiments_false.log \
                               --logger korscibert_lstmcp1 \
                               --model_name roberta_lstmcp1.bin")

    
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 1e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm korscibert \
                               --load_model linearlstm \
                               --logger_file ./data/experiments_false.log \
                               --logger korscibert_linear \
                               --model_name korscibert_linearlstm_1e.bin")
    os.system("python train.py --epochs 10 \
                               --optimizer adamw \
                               --lr 1e-5 \
                               --weight_decay 1e-2 \
                               --eps 1e-8 \
                               --num_warmup_steps 100 \
                               --plm korscibert \
                               --load_model lstmcopy1 \
                               --logger_file ./data/experiments_false.log \
                               --logger korscibert_lstmcp1 \
                               --model_name korscibert_lstmcp1_1e.bin")
