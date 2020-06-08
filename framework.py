from tqdm import trange
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import os
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
        model.zero_grad()
        batch = tuple(t.to(self.args.device) for t in batch_data)

        loss = model.get_loss(
            input_ids=batch[0],
            attention_mask=batch[1],
            labels=batch[-1]
        )
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule
        return loss.item()
        


    def train(self, train_dataset, dev_dataset, model):
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
        best_results = 0
        early_stop = 0

        model.to(self.args.device)
        for epoch in trange(0, int(self.args.num_train_epochs), desc="Epoch", disable=True):
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            for step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch, model)

                if step % 20 == 0:
                    print('Train Epoch[{}] Step[{} / {}] - loss: {:.6f}  '.format(epoch+1, step+1, len(train_dataloader), loss))  # , accuracy, corrects
                global_step += 1
                
                if (self.args.evaluate_step > 0 and global_step % self.args.evaluate_step == 0) or (epoch==int(self.args.num_train_epochs)-1 and step == len(train_dataloader)-1):
                    early_stop += 1
                    results = self.evaluate(dev_dataset, model)
                    print("best result: %.2f, current result: %.2f" % (best_results, results["main"]))
                    if best_results <= results["main"]:
                        best_results = results["main"]
                        print("Saving model checkpoint to %s"%self.args.save_model)
                        torch.save(model.state_dict(), self.args.save_model)
                        early_stop = 0
                    print()
                
                if early_stop >= 5:
                    return

    def evaluate(self, dev_dataset, model):
        print()
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.args.dev_batch_size, shuffle=False)
        model.eval()
        total_loss = 0
        right_count = 0
        right_count_BI = 0
        total_count = 0
        total_count_BI = 0
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                labels = batch[-1]
                loss = model.get_loss(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                    labels=labels
                )
                predicted = model(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                )
                total_loss += loss

                if step % 20 == 0:
                    print('Dev Step[{} / {}] - loss: {:.6f}  '.format(step+1, len(dev_dataloader), loss))

                right_count += ((labels!=-100)&(predicted.eq(labels))).cpu().sum().item()
                total_count += (labels!=-100).cpu().sum().item()
                right_count_BI += ((labels!=-100)&(labels!=0)&(predicted.eq(labels))).cpu().sum().item()
                total_count_BI += ((labels!=-100)&(labels!=0)).cpu().sum().item()

        results = {
            "loss": total_loss / len(dev_dataloader),
            "main": right_count_BI / total_count_BI,
            "acc": right_count / total_count
        }
        print("Dev Loss: {:.6f},  acc: {:.4f}, acc/O: {:.4f}".format(results["loss"], results["acc"], results["main"]))
        return results


    def test(self, test_dataset, model, all_labels):
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.dev_batch_size, shuffle=False)
        model.to(self.args.device)
        model.eval()
        right_count = 0
        predicted_list = []
        labels_list = []
        for step, batch in enumerate(test_dataloader):
            print('Test Step[{} / {}]'.format(step+1, len(test_dataloader)))
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                labels = batch[-1]
                predicted = model(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                )
                predicted_list.append(predicted.cpu())
                labels_list.append(labels.cpu())

        self.metrics(predicted_list, labels_list, all_labels)
        
    def metrics(self, predicted_list, labels_list, all_labels):
        # bs, max_seq_length
        predicted = torch.cat(predicted_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        predicted, labels = predicted.view(-1), labels.view(-1)
        right_count = 0
        right_count_BI = 0
        total_count = 0
        total_count_BI = 0

        right_count += ((labels!=-100)&(predicted.eq(labels))).sum().item()
        total_count += (labels!=-100).sum().item()
        right_count_BI += ((labels!=-100)&(labels!=0)&(predicted.eq(labels))).sum().item()
        total_count_BI += ((labels!=-100)&(labels!=0)).sum().item()
        print("Test acc: {:.4f}, acc/O: {:.4f}".format(right_count / total_count, right_count_BI / total_count_BI))

        num_labels = max(labels) + 1
        predicted = predicted.tolist()
        labels = labels.tolist()
        assert len(predicted) == len(labels)
        metrics = [[0 for i in range(num_labels)] for j in range(num_labels)]

        for i in range(len(labels)):
            if labels[i] != -100:
                metrics[labels[i]][predicted[i]] += 1
        
        # if not os.path.exists(self.args.output_dir):
        #     os.makedirs(self.args.output_dir)
        with open(self.args.output_dir, "w", encoding="utf-8") as f:
            f.write("\t".join(["LABLE"] + all_labels) + "\n")
            for i in range(num_labels):
                f.write("\t".join([all_labels[i]] + list(map(lambda x: str(x), metrics[i]))) + "\n")
        return metrics