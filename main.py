from framework import Framework, set_seed
from processor import NERDataset
from model import BERTforNER
from transformers import BertConfig, BertTokenizer
import argparse
import torch

parser = argparse.ArgumentParser()
# task setting
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--dev_batch_size', type=int, default=6)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--evaluate_step', type=int, default=1000)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=5e-5)

# file path
parser.add_argument('--train_file', type=str, default='./data/train.txt')
parser.add_argument('--dev_file', type=str, default='./data/dev.txt')
parser.add_argument('--test_file', type=str, default='./data/test.txt')
parser.add_argument('--vocab_file', type=str, default='./pretrained/vocab.txt')
parser.add_argument('--save_model', type=str, default='./save_model/')
parser.add_argument('--output_dir', type=str, default='./output/')

# train setting
parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--loss', type=str, default='cross_entropy')

args = parser.parse_args()
# device
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    set_seed(2020)
    # labels
    labels = ['O', 'I-PER', 'B-PER', 'I-LOC', 'I-ORG', 'B-ORG', 'B-LOC']
    args.num_labels = len(labels)

    tokenizer = BertTokenizer(args.vocab_file)
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=args.num_labels, hidden_dropout_prob=args.hidden_dropout_prob)
    model = BERTforNER(config)
    framework = Framework(args)

    if args.mode == "train":
        print("loading datasets...")
        train_dataset = NERDataset(args.train_file, tokenizer, labels, args.max_len)
        dev_dataset = NERDataset(args.dev_file, tokenizer, labels, args.max_len)
        framework.train(train_dataset, dev_dataset, model)

    test_dataset = NERDataset(args.test_file, tokenizer, labels, args.max_len)
    
    print("\nloading model ...")
    model = BERTforNER.from_pretrained(args.save_model)
    framework.test(test_dataset, model)

if __name__ == "__main__":
    main(args)