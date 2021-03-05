import os
import random
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from src.graph_hier_mat_model import HierGraphNet
from src.d_graph_hier_mat_model import DHierGraphNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import time
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

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/test_pair.csv")
    parser.add_argument("--test_set", type=str, default="data/test_pair.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/word_embedding/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_sent_words", type=str, default="5,5")
    parser.add_argument("--graph", type=int, default=0)
    parser.add_argument("--tune", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='whole_model_han')
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # os.chdir("/data2/xuhuizh/graphM_project/HAMN")
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if opt.tune==1:
        print('Fine tune the word embedding')
        freeze = False
    else:
        freeze = True
    if opt.max_sent_words =="5,5":
        max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    else:
        max_word_length, max_sent_length = [int(x) for x in opt.max_sent_words.split(',')]
    print(max_word_length, max_sent_length)
    training_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.word2vec_path, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    if opt.graph==1:
        print('use graph model')
        model = HierGraphNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size,freeze,
                            opt.word2vec_path, max_sent_length, max_word_length)
    elif opt.graph==2:
        print('use deep graph model')
        model = DHierGraphNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size,freeze,
                            opt.word2vec_path, max_sent_length, max_word_length)
    else:
        model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size,freeze,
                       opt.word2vec_path, max_sent_length, max_word_length)
    # if os.path.isdir(opt.log_path):
    #    shutil.rmtree(opt.log_path)
    # os.makedirs(opt.log_path)
    #writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()
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
        for iter, (feature1,feature2, label) in enumerate(training_generator):
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
                loss, training_metrics["accuracy"]),flush=True)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
           # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
           # writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
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
            te_loss, test_metrics["accuracy"]))
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature1, te_feature2, te_label in test_generator:
                num_sample = len(te_label)
                #print(num_sample)
                if torch.cuda.is_available():
                    te_feature1 = te_feature1.cuda()
                    te_feature2 = te_feature2.cuda()
                    te_label = te_label.float().cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature1,te_feature2)
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
           # writer.add_scalar('Test/Loss', te_loss, epoch)
           # writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + opt.model_name)
                logger.info('saved model')

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    localtime = time.asctime( time.localtime(time.time()))
    print(localtime)
    train(opt)
    localtime = time.asctime( time.localtime(time.time()))
    print(localtime)


