import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation
from src.dataset_bert_pla import MyDataset
import argparse
import shutil
import csv
import numpy as np
from src.bert_average import Bert_cls_av
from src.bert_cos import Bert_cos
from src.bert_att import Bert_att
from src.bert_random import Bert_random
from src.bert_han_g import HierAttNet
from src.bert_coss import Bert_coss
from src.bert_han_sg_g import HierGraphAttNet 

models_class = {'Bert_avg':Bert_cls_av, 
'Bert_han_g':HierAttNet, 
'Bert_han_sg_g':HierGraphAttNet,
'Bert_cos':Bert_cos,
'Bert_coss':Bert_coss,
'Bert_random':Bert_random,
'Bert_att':Bert_att}

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="data/cite_acl/test_cite.csv")
    parser.add_argument("--pos_path", type=str, default="data/cite_acl/test_cite.csv")
    parser.add_argument("--pre_trained_model", type=str, default="acl_bert_model/Bert_han_sg_g.pth")
    parser.add_argument("--word2vec_path", type=str, default="data/word_embedding/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50) 
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--model_type", type=str, default='Bert_han_sg_g')
    args = parser.parse_args()
    return args

def masking(pred_pos, mask):
    pos_list = []
    if mask==[]:
        return pred_pos
    pred_pos = pred_pos.transpose(0,1)
    pred_pos = pred_pos.numpy()
    mask = [i for index, i in enumerate(mask) if index%2==0]
    for i, j in zip(pred_pos, mask):
        pos = np.multiply(i,j)
        pos = torch.from_numpy(pos)
        pos = pos.unsqueeze(dim=1)
        pos_list.append(pos)
    pos_list = torch.cat(pos_list, dim=1)
    return pos_list
        
def pos_cal_mpr(pred_pos, mask, true_pos, label_list, pred_label):
    count = 0
    mpr = 0
    pred_pos = torch.cat(pred_pos, dim=1)
    pred_pos = masking(pred_pos, mask)
    _, pred_doc_pos = pred_pos.sort(dim=0,descending=True)
    pred_doc_pos = pred_doc_pos.transpose(0,1)
    pred_doc_pos = pred_doc_pos.numpy()
    pred_doc_pos = list(pred_doc_pos)
    for pred, true, label, pre_label in zip(pred_doc_pos, true_pos, label_list, pred_label):
        if label!=0 and pre_label!=0 and true!=[]: 
            rank = []  
            for i in true:
                #print(list(pred))
                try:
                    rank.append(list(pred).index(i)+1)
                except:
                    rank.append(60)
            rank_f = min(rank)     
            mpr += 1/rank_f
            count += 1
        elif pre_label==label:
            mpr += 1
            count += 1
        else:
            mpr += 1/60
            count += 1
    return mpr/count

def pos_accuracy(pred_pos, mask, true_pos, label_list,pred_label, top_num):
    true_count = 0
    count = 0
    #print(label_list.shape)
    pred_pos = torch.cat(pred_pos, dim=1)
    pred_pos = masking(pred_pos, mask)
    #print(pred_pos.shape)
    print(pred_pos.transpose(0,1)[6:10])
    _, pred_doc_pos = (pred_pos).topk(top_num,dim=0)
    pred_doc_pos = pred_doc_pos.transpose(0,1)
    pred_doc_pos = pred_doc_pos.numpy()
    pred_doc_pos = list(pred_doc_pos)
    #print(true_pos[6:20])
    #print(pred_doc_pos[6:20])
    #print(label_list[6:20])
    for pred, true, label, pre_label in zip(pred_doc_pos, true_pos, label_list, pred_label):
        if label!=0 and pre_label!=0:
            pred = set(pred)
            true = set(true)
            if pred.intersection(true) != set():
                true_count += len(list(pred.intersection(true))) 
                count += top_num 
            else:
                count += top_num 
        elif label==pre_label:
            true_count += 1
            count += 1
        else:
            count += 1
    return true_count/count

def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        freeze=True
        model = models_class[opt.model_type](vector_size=1024) 
        model.load_state_dict(torch.load(opt.pre_trained_model))
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)
    test_set = MyDataset(opt.data_path, opt.pos_path, opt.max_length)
    pos = test_set.get_pos()
    test_generator = DataLoader(test_set, **test_params)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    te_pos_ls = []
    te_true_post_ls = []
    for te_feature1, te_feature2, te_label, _ in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature1 = te_feature1.cuda()
            te_feature2 = te_feature2.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature1, te_feature2)
            doc_te_predictions = te_predictions[-1]
            pos_predictions = te_predictions[:-1]
            #te_predictions = F.softmax(te_predictions) #do not know what it is doing?
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(doc_te_predictions.clone().cpu())
        te_pos_ls.append(pos_predictions.clone().cpu())
        #break 
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)
    print(te_pred)
    te_pred = np.where(te_pred > 0.5, 1, 0)
    mask = test_set.get_mask() 
    pos_acc_10 = pos_accuracy(te_pos_ls,mask, pos, te_label, te_pred, 10)
    pos_acc_5 = pos_accuracy(te_pos_ls, mask, pos, te_label, te_pred, 5)
    pos_acc_1 = pos_accuracy(te_pos_ls, mask, pos, te_label, te_pred, 1)
    pos_mpr = pos_cal_mpr(te_pos_ls, mask, pos, te_label, te_pred)
    fieldnames = ['True label', 'Predicted label', 'Content1', 'Content2']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(te_label, te_pred, test_set.texts):
            writer.writerow(
                {'True label': i, 'Predicted label':j , 'Content1': k[0], 'Content2':k[1]})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix", "f1"])
    print("Prediction:\nLoss: {} Accuracy: {} F1: {} Pos Acc 10: {} Pos Acc 5:{} Pos Acc 1:{}  mpr: {}\nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["f1"],
                                                                               pos_acc_10,pos_acc_5,pos_acc_1,
                                                                               pos_mpr,
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
