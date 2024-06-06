import re
from numpy import *
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import KFold
import random
import os
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from data_loader import GetLoader, GetLoader1, GetLoader2
from torch.autograd import Function
from torch import optim
from pytorchtools import EarlyStopping
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
# from model import CNNModel,Dense5Model
from sklearn import svm, metrics, preprocessing
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from lmmd import MMDLoss, LMMD_loss
from models import *
from pre_data import seed_data


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy




def train(model, device, src_dataloader, target_dataloader, criterion, num_epoch, model_path, bs, scheduler=None):
    """
    Implementation of training process.
    See more high-level details on http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
    :param model: model which we will train
    :param device: cpu or gpu
    :param src_dataloader: dataloader with source images
    :param target_dataloader:  dataloader with target images
    :param criterion: loss function
    :param optimizer: method for update weights in NN
    :param epoch: current epoch
    :param scheduler: algorithm for changing learning rate
    """
    len_source = len(src_dataloader)
    len_target = len(target_dataloader)
    len_dataloader = min(len_source, len_target)
    valid_losses = []

    temperature = 4
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)
    LR = 0.001
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # model.train()
    for epoch in range(num_epoch):
        # alpha=1
        # print(alpha, LR)
        iter_source = iter(src_dataloader)
        iter_target = iter(target_dataloader)
        model.train()
        for batch_idx in range(len_dataloader):

            if batch_idx % len_source == 0:
                iter_source = iter(src_dataloader)

            if batch_idx % len_target == 0:
                iter_target = iter(target_dataloader)
            imgs_src, eye_src, src_class = iter_source.next()
            imgs_target, eye_target, _ = iter_target.next()

            # if len(imgs_target) != len(imgs_src):
            #     imgs_src=imgs_src[:len(imgs_target)]
            #     eye_src = eye_src[:len(imgs_target)]
            #     src_class=src_class[:len(imgs_target)]

            imgs_src, src_class = imgs_src.to(device), src_class.to(device)
            eye_src = eye_src.to(device)
            imgs_target = imgs_target.to(device)
            eye_target = eye_target.to(device)
            # alpha=1

            model.zero_grad()

            # train on target domain
            t_class_predict, t_feature_eeg, t_feature_eye, t_class_predict_eeg, t_class_predict_eye = model(imgs_target,
                                                                                                            eye_target)

            t_class_predict = t_class_predict_eeg + t_class_predict_eye
            t_class_predict_temp = t_class_predict / temperature
            target_softmax_out_temp = nn.Softmax(dim=1)(t_class_predict_temp)
            target_entropy_weight = Entropy(target_softmax_out_temp).detach()
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
            target_entropy_weight = bs * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                target_softmax_out_temp)
            # cov_matrix_t = target_softmax_out_temp.transpose(1, 0).mm(target_softmax_out_temp)
            cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
            mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 3
            ''''''

            t_label = F.softmax(t_class_predict_eeg, dim=1).detach()
            t_label_eye = F.softmax(t_class_predict_eye, dim=1).detach()
            eeg_entropy = Entropy(t_label).detach()
            eye_entropy = Entropy(t_label_eye).detach()
            t_label[eeg_entropy > eye_entropy, :] = t_label_eye[eeg_entropy > eye_entropy, :]
            # t_label[t_label.max(1)[0] < t_label_eye.max(1)[0], :] = t_label_eye[t_label.max(1)[0] < t_label_eye.max(1)[0], :]
            t_label = t_label.detach()

            # entropy = Entropy(t_label).detach()
            # weight = 1 + torch.exp(-entropy)
            # weight = bs * weight / torch.sum(weight)
            # mseloss1 = (torch.mean((F.softmax(t_class_predict_eeg, dim=1)-t_label)**2,dim = 1)).mean()
            # mseloss2 = (torch.mean((F.softmax(t_class_predict_eye, dim=1)-t_label)**2,dim = 1)).mean()
            # mseloss = mseloss1 + mseloss2

            # train on source domain
            class_predict, s_feature_eeg, s_feature_eye, class_predict_eeg, class_predict_eye = model(imgs_src, eye_src)
            # src_class_loss = criterion(class_predict, src_class)
            src_eeg_loss = criterion(class_predict_eeg, src_class)
            src_eye_loss = criterion(class_predict_eye, src_class)
            criterion1 = nn.KLDivLoss(reduction='batchmean')
            criterion2 = nn.MSELoss()
            criterion3 = MMDLoss()
            criterion4 = LMMD_loss(class_num=4, device=device)

            # class_predict = nn.LogSoftmax(dim=1)(class_predict)
            kl_loss = criterion1(nn.LogSoftmax(dim=1)(t_class_predict_eeg),
                                 nn.Softmax(dim=1)(t_class_predict_eye).clone().detach()) \
                      + criterion1(nn.LogSoftmax(dim=1)(t_class_predict_eye),
                                   nn.Softmax(dim=1)(t_class_predict_eeg).clone().detach())

            # kl_loss1 = criterion1(F.log_softmax(t_class_predict_eeg, dim=1),t_label)
            # kl_loss2 = criterion1(F.log_softmax(t_class_predict_eye, dim=1),
            #                       t_label)
            #mseloss = criterion2(F.softmax(t_class_predict_eeg, dim=1), F.softmax(t_class_predict_eye, dim=1))
            mseloss = criterion2(F.softmax(t_class_predict_eeg, dim=1), t_label) + criterion2(
                F.softmax(t_class_predict_eye, dim=1), t_label)
            # s_feature_loss = get_L2norm_loss_self_driven(s_feature)
            # t_feature_loss = get_L2norm_loss_self_driven(t_feature)
            # mmd_loss = criterion3(s_feature_eeg,t_feature_eeg)+criterion3(s_feature_eye,t_feature_eye)
            mmd_loss = criterion3(torch.cat((s_feature_eeg, s_feature_eye), dim=-1),
                                  torch.cat((t_feature_eeg, t_feature_eye), dim=-1))
            # lmmd_loss = criterion4(torch.cat((s_feature_eeg, s_feature_eye), dim=-1),torch.cat((t_feature_eeg, t_feature_eye), dim=-1), src_class,
            #                        0.5*nn.Softmax(dim=1)(t_class_predict_eeg)+0.5*nn.Softmax(dim=1)(t_class_predict_eye))
            # lmmd_loss = 0.5*criterion4(s_feature_eeg, t_feature_eeg, src_class, F.softmax(t_class_predict_eeg, dim=1))\
            #             + 0.5*criterion4(s_feature_eye, t_feature_eye, src_class, F.softmax(t_class_predict_eye, dim=1))

            # calculating loss
            # loss = src_class_loss + (src_domain_loss + t_domain_loss)
            # loss = src_class_loss + mcc_loss + s_feature_loss + t_feature_loss
            # loss = src_class_loss + 1*mcc_loss
            # loss = src_class_loss
            lambd = 2 / (1 + math.exp(-10 * (epoch + 1) / num_epoch)) - 1
            loss = 1 * src_eeg_loss + 1 * src_eye_loss+mseloss+mmd_loss
            # print('err_s_label: %f, err_s_domain: %f, err_t_domain: %f,loss: %f' \
            # % (src_class_loss.data.cpu().numpy(),src_domain_loss.data.cpu().numpy(),
            # t_domain_loss.data.cpu().item(),loss.data.cpu().numpy()))
            if scheduler is not None:
                scheduler.step()

            """
            Calculating gradients and update weights
            """
            loss.backward()
            optimizer.step()

            valid_losses.append(loss.item())

        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        '''
        model.eval()
        accuracy = 0
        accuracy_domain = 0
        for (imgs, labels) in target_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction, domain_prediction = model(imgs, alpha=0)
            domain_labels = torch.ones(len(labels)).long().to(device)
            pred_cls = prediction.data.max(1)[1]
            pred_domain = domain_prediction.data.max(1)[1]
            accuracy += pred_cls.eq(labels.data).sum().item()
            accuracy_domain += pred_domain.eq(domain_labels.data).sum().item()


        accuracy /= len(target_dataloader.dataset)
        accuracy_domain /= len(target_dataloader.dataset)
        print(f'Accuracy on SEED-test: {100 * accuracy:.2f}%')
        print(f'accuracy_domain on SEED-test: {100 * accuracy_domain:.2f}%')
        '''


def test(model, device, test_loader):
    """
    Provide accuracy on test dataset
    :param model: Model of the NN
    :param device: cpu or gpu
    :param test_loader: loader of the test dataset
    :param max: the current max accuracy of the model
    :return: max accuracy for overall observations
    """
    model.eval()

    accuracy = 0
    accuracy_domain = 0
    with torch.no_grad():
        for (imgs, eyes, labels) in test_loader:
            imgs = imgs.to(device)
            eyes = eyes.to(device)
            labels = labels.to(device)
            prediction, _, _, prediction_eeg, prediction_eye = model(imgs, eyes)
            # prediction_all = prediction + prediction_eeg + prediction_eye
            prediction = prediction_eeg + prediction_eye
            pred_cls = prediction.data.max(1)[1]
            accuracy += pred_cls.eq(labels.data).sum().item()

    accuracy /= len(test_loader.dataset)

    return accuracy


'''

def plot_with_labels(lowDWeights, labels):
    plt.cla() #clear当前活动的坐标轴
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.text(x, y, str(s),color=c,fontdict={'weight': 'bold', 'size': 9}) #在指定位置放置文本
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.savefig('1.png')
    plt.show()

def test(model, device, test_loader,source_loader):
    """
    Provide accuracy on test dataset
    :param model: Model of the NN
    :param device: cpu or gpu
    :param test_loader: loader of the test dataset
    :param max: the current max accuracy of the model
    :return: max accuracy for overall observations
    """
    model.eval()

    accuracy = 0
    data_zip = enumerate(zip(source_loader, test_loader))
    plt.ion()

    for batch_idx, ((imgs_src, src_labels), (imgs, labels)) in data_zip:
        d_src = torch.zeros(len(imgs_src)).long().to(device)
        d_target = torch.ones(len(imgs)).long().to(device)
        imgs_src = imgs_src.to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction, feature = model(imgs)
        prediction_src, feature_src = model(imgs_src)
        pred_cls = prediction.data.max(1)[1]
        accuracy += pred_cls.eq(labels.data).sum().item()

        if batch_idx % 100 == 0:
            # t-SNE 是一种非线性降维算法，非常适用于高维数据降维到2维或者3维，进行可视化
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # 最多只画500个点
            plot_only = 500
            # fit_transform函数把last_layer的Tensor降低至2个特征量,即3个维度(2个维度的坐标系)
            low_dim_embs = tsne.fit_transform(feature.cpu().data.numpy()[:plot_only, :])
            labels = labels.cpu().numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)

    accuracy /= len(test_loader.dataset)

    return accuracy
'''

def get_data(sub_id, eeg_data, eye_data, label):
    test_x = eeg_data[sub_id]
    train_x = np.delete(eeg_data,sub_id,axis=0)
    train_x = train_x.reshape((-1,train_x.shape[-1]))
    test_x_eye = eye_data[sub_id]
    train_x_eye = np.delete(eye_data, sub_id, axis=0)
    train_x_eye = train_x_eye.reshape((-1, train_x_eye.shape[-1]))
    test_y = label[sub_id]
    train_y = np.delete(label, sub_id, axis=0)
    train_y = train_y.reshape((-1))
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

    return train_x, test_x, train_y, test_y, train_x_eye, test_x_eye

if __name__ == '__main__':
    # RNN:t=4 bs=100 lanmuda=2
    # MLP:t=4 bs=200 lanmuda=1
    """
    Set up random seed for reproducibility
    """
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Set up number of epochs, domain threshold, loss function, device
    """
    EPOCHS = 200
    BATCH_SIZE = 100
    criterion = nn.CrossEntropyLoss()
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    acc = []
    pred_all = []
    y_all = []

    eeg_data, eye_data, label = seed_data()



    for sub_id in range(12):
        print("processing ", sub_id)
        train_x, test_x, train_y, test_y, train_x_eye, test_x_eye = get_data(sub_id, eeg_data, eye_data, label)

        if np.isnan(train_x_eye).any():
            train_x_eye[np.isnan(train_x_eye)] = 0
        if np.isnan(test_x_eye).any():
            test_x_eye[np.isnan(test_x_eye)] = 0
        '''
        # shuffle data
        index = np.array(range(0, len(train_y)))
        np.random.shuffle(index)
        train_x = train_x[index][:len(test_y)]
        train_y = train_y[index][:len(test_y)]
        train_x_eye = train_x_eye[index][:len(test_y)]
        print(train_x.shape)
        print(train_y.shape)

        cla = svm.SVC(kernel='linear')
        # cla.fit(train_x_eye, train_y)
        # pred = cla.predict(test_x_eye)
        cla.fit(train_x, train_y)
        pred = cla.predict(test_x)
        accuracy = np.sum(pred == test_y) / len(pred)

        pred_all.append(pred)
        y_all.append(test_y)

        '''

        dataset_source = GetLoader2(train_x, train_x_eye, train_y)
        dataset_target = GetLoader2(test_x, test_x_eye, test_y)
        source_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        target_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        """
        Create a model and send it to device. After create an optimizer and scheduler
        """
        model = M3TNet_SEED(k=4).to(device)
        #model = MLPModel().to(device)

        """
        Training loop
        """
        accuracy = 0
        model_path = 'models/best_model{0}.pt'.format(sub_id)
        train(model, device, source_dataloader, target_dataloader, criterion, EPOCHS,model_path,BATCH_SIZE)
        print('----------------------------------------')
        del model

        """
        Evaluating of the model by loading the best weights after training.
        """

        model = M3TNet_SEED(k=4)
        #model = MLPModel()
        print(model)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        accuracy = test(model, device, target_dataloader)
        del model

        acc.append(accuracy)
        print(acc)
    acc = np.array(acc)
    print(np.mean(acc))

    # pred_all = np.array(pred_all).reshape((-1))
    # y_all = np.array(y_all).reshape((-1))
    # C = confusion_matrix(y_all, pred_all)
    # print(C)