# Chinese-NER
Chinese Named Entity Recognition (中文命名实体识别).


## Update
**Warning**: When the training loss drops but the evaluation metrics are 0, please check whether the suitable learning rate is selected for Bert and Bi-LSTM respectively.

### 2020.06.30 
* Fixed a bug that failed to load the pre-trained model.
* Use the representation of the first sub-token after the wordpiece for classification.

## Environment

* python>=3.6.4
* pytorch==1.5.0
* transformers==2.11.0
* seqeval==0.0.12
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

We use the renminribao2014 (人民日报2014) dataset (https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA password: 1fa3). Download and unzip it in `data/renminribao2014/`, then run `process_rmrb.py`.
```
python process_rmrb.py 
```
Modify the `labels` in `main.py` according to your dataset:
```
labels = ['O', 'B-LOC', 'B-ORG', 'B-T', 'I-LOC', 'I-PER', 'B-PER', 'I-ORG', 'I-T']
```

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
# set the learning rate to 1e-2
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

## Results
Report on the subset of renminribao2014 due to the limitation of computing resources. The size of subset is as follows:

Training Set|Validation set|Test set
|:-:|:-:|:-:|
20,000|1,000|1,000|

**BiLSTM**

|label|  precision|  recall|  f1-score|  support
| :-:     | :-:     | :-:     | :-:     | :-:     |
|  PER|  0.7679|  0.8240|  0.7950|  767
|  LOC|  0.8119|  0.8643|  0.8373|  884
|  ORG|  0.8116|  0.7671|  0.7887|  73
|  T|  0.9157|  0.9263|  0.9210|  868
**micro avg**|  0.8322|  0.8704|  0.8508|  2592
**macro avg**|  0.8336|  0.8704|  0.8514|  2592

**BiLSTM + CRF**

|label| precision| recall | f1-score| support
| :-:     | :-:     | :-:     | :-:     | :-:     |
| PER| 0.8780| 0.8631| 0.8705| 767
| LOC| 0.9172| 0.8903| 0.9036| 884
| ORG| 0.9167| 0.9041| 0.9103| 73
| T| 0.9661| 0.9516| 0.9588| 868
**micro avg**| 0.9220| 0.9032| 0.9125| 2592
**macro avg**| 0.9220| 0.9032| 0.9125| 2592

**BERT**
|label| precision| recall | f1-score| support
| :-:     | :-:     | :-:     | :-:     | :-:     |
| PER| 0.9281| 0.9257| 0.9269| 767
| LOC| 0.9224| 0.9422| 0.9322| 883
| ORG| 0.9324| 0.9452| 0.9388| 73
| T| 0.9634| 0.9700| 0.9667| 868
**micro avg**| 0.9380| 0.9467| 0.9424| 2591
**macro avg**| 0.9381| 0.9467| 0.9424| 2591

**BERT + CRF**
|label| precision| recall | f1-score| support
| :-:     | :-:     | :-:     | :-:     | :-:     |
| PER| 0.9396| 0.9322| 0.9359| 767
| LOC| 0.9357| 0.9558| 0.9457| 883
| ORG| 0.9211| 0.9589| 0.9396| 73
| T| 0.9622| 0.9689| 0.9656| 868
**micro avg**| 0.9453| 0.9533| 0.9493| 2591
**macro avg**| 0.9453| 0.9533| 0.9493| 2591

## References

* **pytorch CRF**: https://github.com/kmkurn/pytorch-crf
