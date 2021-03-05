import os
import random
import torch
import torch.nn as nn
from transformers import *
from torch.utils.data import DataLoader
from sklearn import metrics
from src.utils import get_max_lengths, get_evaluation
from src.bert_average import Bert_cls_av
from src.bert_han_g import HierAttNet
from src.bert_sg_average import Bert_sg_av 
from src.bert_han_sg_g import HierGraphAttNet 
from src.bert_han_dg_g import DHierGraphNet
from src.dataset_bert_3 import MyDataset
import argparse
import shutil
import numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

models_class = {'Bert_avg':Bert_cls_av,
'Bert_han_g':HierAttNet, 
'Bert_han_sg_g':HierGraphAttNet,
'Bert_han_dg_g':DHierGraphNet,
'Bert_sg_avg':Bert_sg_av}
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/test_pair.csv")
    parser.add_argument("--test_set", type=str, default="data/test_pair.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--graph", type=int, default=0)
    parser.add_argument("--tune", type=int, default=1)
    parser.add_argument("--model_type", type=str, default='Bert_average')
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=18)
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    #os.chdir("/data2/xuhuizh/graphM_project/HAMN")
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
 
    training_set = MyDataset(opt.train_set, opt.max_len)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.max_len)
    test_generator = DataLoader(test_set, **test_params)
    model = models_class[opt.model_type](vector_size=1024, sent_hidden_size=opt.hidden_size, batch_size=opt.batch_size)
    #if os.path.isdir(opt.log_path):
    #    shutil.rmtree(opt.log_path)
    #os.makedirs(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()
    #m = nn.Sigmoid()
    #criterion = nn.CosineEmbeddingLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        start_time = time.time()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for iter, (feature1, feature2, label, _) in enumerate(training_generator):
            num_sample = len(label)
            if torch.cuda.is_available():
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                label = label.float().cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature1, feature2)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            loss_ls.append(loss * num_sample)
            te_label_ls.extend(label.clone().cpu())
            te_pred_ls.append(predictions.clone().cpu())
            sum_all = 0
            sum_updated = 0
            '''
            for name, param in model.named_parameters():
                print('All parameters')
                print(name,torch.numel(param.data))
                sum_all += torch.numel(param.data)
                if param.requires_grad:
                    print('Updated parameters:')
                    print(name,torch.numel(param.data))
                    sum_updated+= torch.numel(param.data)
            print('all', sum_all)
            print('update', sum_updated)
            '''
        #print total train loss
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.detach().numpy(), list_metrics=["accuracy", "confusion_matrix"])
        output_file.write(
            "Epoch: {}/{} \nTrain loss: {} Train accuracy: {} \nTrain confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epoches,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epoches,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]),flush=True)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature1, te_feature2, te_label, _ in test_generator:
                num_sample = len(te_label)
                #print(num_sample)
                if torch.cuda.is_available():
                    te_feature1 = te_feature1.cuda()
                    te_feature2 = te_feature2.cuda()
                    te_label = te_label.float().cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature1, te_feature2)
                te_predictions = te_predictions[-1]
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name=='fd.weight':
                        print(name,param.data)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model.state_dict(), opt.saved_path + os.sep + opt.model_type+'.pth') 

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break

if __name__ == "__main__":
    opt = get_args()
    train(opt)
