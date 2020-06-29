from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import os
from seqeval.metrics import f1_score, classification_report
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2020)

class Framework:

    def __init__(self, args):
        self.args = args
 
    def train_step(self, batch_data, model):
        model.train()
        batch = tuple(t.to(self.args.device) for t in batch_data)
        input_ids, attention_mask, pred_mask, labels = batch
        predicted, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pred_mask=pred_mask,
            input_labels=labels
        )
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        return loss.item()
        

    def train(self, train_dataset, dev_dataset, model, labels):
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)

        # get optimizer schedule and loss
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(train_dataloader) * self.args.num_train_epochs
        )
        
        # Train!
        print("***** Running training *****")
        print("  Num examples = ", len(train_dataset))
        print("  Num Epochs = ", self.args.num_train_epochs)

        global_step = 0
        # Check if continuing training from a checkpoint
        best_result = 0
        early_stop = 0

        model.to(self.args.device)
        for epoch in range(0, int(self.args.num_train_epochs)):
            for step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch, model)
                if step % 20 == 0:
                    print('Train Epoch[{}] Step[{} / {}] - loss: {:.6f}  '.format(epoch+1, step+1, len(train_dataloader), loss))  # , accuracy, corrects
                global_step += 1
                
                if (self.args.evaluate_step > 0 and global_step % self.args.evaluate_step == 0) or (epoch==int(self.args.num_train_epochs)-1 and step == len(train_dataloader)-1):
                    early_stop += 1
                    result = self.evaluate(dev_dataset, model, labels)
                    print("best f1: %.2f, current f1: %.2f" % (best_result, result))
                    if best_result <= result:
                        best_result = result
                        print("Saving model checkpoint to %s"%self.args.save_model)
                        torch.save(model.state_dict(), self.args.save_model)
                        early_stop = 0
                    print()
                
                if early_stop >= 5:
                    return

    def evaluate(self, dev_dataset, model, all_labels):
        print("\n Evaluating ...")
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.args.dev_batch_size, shuffle=False)
        model.eval()
        total_loss = 0        
        predicted_list = []
        labels_list = []
        for step, batch in enumerate(tqdm(dev_dataloader)):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, pred_mask, labels = batch
                predicted, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pred_mask=pred_mask,
                    input_labels=labels
                )
                total_loss += loss.item()
                
                if self.args.crf:
                    predicted = [seq[seq>=0].tolist() for seq in predicted]
                else:
                    predicted = [seq[mask==1].tolist() for seq, mask in zip(predicted, pred_mask)]

                groud_labels = [seq[mask==1].tolist() for seq, mask in zip(labels, pred_mask)]

                for tl, pl in zip(groud_labels, predicted):
                    labels_list.append([all_labels[l] for l in tl])
                    predicted_list.append([all_labels[l] for l in pl])
        
        print("Dev Loss: {:.6f}".format(total_loss / len(dev_dataloader)))
        class_report = classification_report(labels_list, predicted_list, digits=4)
        print(class_report)      

        return f1_score(labels_list, predicted_list)


    def test(self, test_dataset, model, all_labels):

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.dev_batch_size, shuffle=False)
        model.to(self.args.device)
        model.eval()
        predicted_list = []
        labels_list = []
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, pred_mask, labels = batch
                predicted = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pred_mask=pred_mask
                )
                predicted = predicted[0]

                if self.args.crf:
                    predicted = [seq[seq>=0].tolist() for seq in predicted]
                else:
                    predicted = [seq[mask==1].tolist() for seq, mask in zip(predicted, pred_mask)]

                groud_labels = [seq[mask==1].tolist() for seq, mask in zip(labels, pred_mask)]

                for tl, pl in zip(groud_labels, predicted):
                    labels_list.append([all_labels[l] for l in tl])
                    predicted_list.append([all_labels[l] for l in pl])
                
        with open(self.args.output_dir, "w", encoding="utf-8") as f:
            for labels in predicted_list:
                for l in labels:
                    f.write(l+"\n")
                f.write("\n")
        
        class_report = classification_report(labels_list, predicted_list, digits=4)
        print(class_report)      
        with open(self.args.output_dir+"_report", "w", encoding="utf-8") as f:
            f.write(class_report)