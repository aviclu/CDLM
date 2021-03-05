import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation
from src.dataset import MyDataset
import argparse
import shutil
import csv
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="data/cite_acl/test_cite.csv")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/model_han_dg")
    parser.add_argument("--word2vec_path", type=str, default="data/word_embedding/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--max_sent_words", type=str, default="33,10")
    args = parser.parse_args()
    return args


def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    max_word_length, max_sent_length = [int(x) for x in opt.max_sent_words.split(',')]
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        model = torch.load(opt.pre_trained_model)
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)
    test_set = MyDataset(opt.data_path, opt.word2vec_path, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    for te_feature1, te_feature2, te_label in test_generator:
        num_sample = len(te_label)
        print(num_sample)
        if torch.cuda.is_available():
            te_feature1 = te_feature1.cuda()
            te_feature2 = te_feature2.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature1, te_feature2)
            #te_predictions = F.softmax(te_predictions) #do not know what it is doing?
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)
    te_pred = np.where(te_pred > 0.5, 1, 0)

    fieldnames = ['True label', 'Predicted label', 'Content1', 'Content2']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(te_label, te_pred, test_set.texts):
            writer.writerow(
                {'True label': i, 'Predicted label':j , 'Content1': k[0], 'Content2':k[1]})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix", "f1"])
    print("Prediction:\nLoss: {} Accuracy: {} F1: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["f1"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
