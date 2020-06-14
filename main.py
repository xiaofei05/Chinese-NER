from framework import Framework, set_seed
from processor import Tokenizer, NERDataset
from model import BERTforNER_CRF, BiLSTM_CRF
from transformers import BertConfig, BertTokenizer
import argparse
import torch
import os
parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--model', type=str, default='bert', choices=['bert', 'bilstm'])
parser.add_argument('--crf', action='store_true')

# train setting
parser.add_argument('--evaluate_step', type=int, default=1000)
parser.add_argument('--max_len', type=int, default=128)

parser.add_argument('--train_batch_size', type=int, default=12)
parser.add_argument('--dev_batch_size', type=int, default=6)
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=5e-5)
# parser.add_argument('--train_batch_size', type=int, default=64)
# parser.add_argument('--dev_batch_size', type=int, default=32)
# parser.add_argument('--num_train_epochs', type=int, default=20)
# parser.add_argument('--learning_rate', type=float, default=1e-3)

# for BiLSTM
parser.add_argument('--embedding_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=200)


# file path
parser.add_argument('--train_file', type=str, default='./data/train.txt')
parser.add_argument('--dev_file', type=str, default='./data/dev.txt')
parser.add_argument('--test_file', type=str, default='./data/test.txt')
parser.add_argument('--save_model', type=str, default='./save_model/')
parser.add_argument('--output_dir', type=str, default='./output/')

# others
parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)

args = parser.parse_args()

# device
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for file_dir in [args.save_model, args.output_dir]:
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

if args.crf:
    save_name = args.model + "_crf"
else:
    save_name = args.model

args.save_model = os.path.join(args.save_model, save_name + ".pt")
args.output_dir = os.path.join(args.output_dir, save_name + ".metrics")

def main(args):
    set_seed(2020)
    labels = ['O', 'B_LOC', 'B_ORG', 'B_T', 'I_LOC', 'I_PER', 'B_PER', 'I_ORG', 'I_T']
    # labels = ['O', 'I-PER', 'B-PER', 'I-LOC', 'I-ORG', 'B-ORG', 'B-LOC']
    args.num_labels = len(labels)
    
    if args.model == 'bert':
        # use 'bert-base-chinese' model
        pretrained_model_name = 'bert-base-chinese'
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        config = BertConfig.from_pretrained(pretrained_model_name, num_labels=args.num_labels, hidden_dropout_prob=args.hidden_dropout_prob)
        config.name = pretrained_model_name

        add_CLS = True
        model = BERTforNER_CRF(config, use_crf=args.crf)
    else:
        
        add_CLS = False
        tokenizer = Tokenizer(args.train_file)
        model = BiLSTM_CRF(len(tokenizer), args.embedding_dim, args.hidden_dim, args.num_labels, args.hidden_dropout_prob, args.crf)

    framework = Framework(args)

    if args.mode == "train":
        print("loading datasets...")
        train_dataset = NERDataset(args.train_file, tokenizer, labels, args.max_len, add_CLS)
        dev_dataset = NERDataset(args.dev_file, tokenizer, labels, args.max_len, add_CLS)
        framework.train(train_dataset, dev_dataset, model)
    
    test_dataset = NERDataset(args.test_file, tokenizer, labels, args.max_len, add_CLS)

    print("\nloading model ...")

    model.load_state_dict(torch.load(args.save_model))
    framework.test(test_dataset, model, labels)

if __name__ == "__main__":
    print(args)
    main(args)
