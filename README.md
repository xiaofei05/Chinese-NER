# Chinese-NER
Chinese Named Entity Recognition (中文命名实体识别).

## Environment

* python>=3.6
* pytorch>=1.0.0
* transformers==2.11.0
* tqdm

## Model

* BiLSTM
* BiLSTM + CRF
* BERT
* BERT + CRF

## Preparation
### Data Format

Each line contains a character and its label, separated by "\t" or space. Each sentence is followed by a blank line.

```
中	B-LOC
国	I-LOC
很	O
大	O

句	O
子	O
结	O
束	O
是	O
空	O
行	O
```

We use the renminribao2014 (人民日报2014) dataset (https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA password: 1fa3). Download and unzip it in `data/`, then run `data/process_rmrb.py`.
```
python data/process_rmrb.py 
```
Modify the `labels` in `main.py` according to your dataset:
```
labels = ['O', 'B_LOC', 'B_ORG', 'B_T', 'I_LOC', 'I_PER', 'B_PER', 'I_ORG', 'I_T']
```
**Remember that label 'O' should be the first.**

## Usage
### **Train**
```
# run bert
python main.py --model bert

# run bert+crf
python main.py --model bert --crf
```

```
# run bilstm
# it's better to reset the parameters
python main.py --model bilstm \
        --learning_rate 1e-2 \
        --num_train_epochs 20 \
        --train_batch_size 64 \
        --dev_batch_size 32

# run bilstm+crf
python main.py --model bilstm --crf \
        --learning_rate 1e-2 \
        --num_train_epochs 20 \
        --train_batch_size 64 \
        --dev_batch_size 32
```
### **Test**

```
# run bert
python main.py --model bert --mode test

# run bert+crf
python main.py --model bert --crf --mode test

# run bilstm
python main.py --model bilstm --mode test

# run bert+crf
python main.py --model bilstm --crf --mode test
```

## References

* https://github.com/lonePatient/BERT-NER-Pytorch