import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import random
from utils import save_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.optim import lr_scheduler
from Bias_net import Generator
from Clus_net import ClusterNet
from data_loader import get_ids, get_images_ids
from pre_processed import batch
from bias_loss import BiasPredictLoss
from clus_loss import ClusterLoss
from tv_loss import TVLoss
from regularizer_loss import RegularizerLoss

epochs_x = []
train_loss_bias = []
train_loss_clus = []

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def rot90(img, direction):
    img = img.transpose(2,3)
    size = img.shape[3]
    inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
    if(direction == 0):
        imgRot = torch.index_select(img, 3, inv_idx) #clock-wise
    else:
        imgRot = torch.index_select(img, 2, inv_idx) #inv clock-wise

    return imgRot

def flip(img):
    size = img.shape[3]
    inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
    imgFlp = torch.index_select(img, 2, inv_idx)
    return imgFlp

def invTrans(img, num):
    if(num == 2):
        return flip(img)
    else:
        return rot90(img,1 - num)

def Trans(img, num):
    if(num == 2):
        return flip(img)
    else:
        return rot90(img,num)

def train_fn(gen_C, clu_C, orimg_path, ori_ids, opt_gen, opt_clu, BIAS, CLUS, REG, TV,
                 lr_gen, lr_clu):

    min_clus_loss = 9999
    min_bias_loss = 9999
    print(
        'Training parameters(cos):Model:{}, Epochs:{} Batch size:{} Learning rate:{}'.format(config.model, config.epochs, config.batchSize, config.learningRate))

    for epoch in range(config.epochs):
        sum_loss_bias = 0
        sum_loss_clus = 0
        sum_loss_inte = 0
        train_num = len(ori_ids)
        random.shuffle(ori_ids)
        train_imgs = get_images_ids(ori_ids, orimg_path)
        ####################
        for j, b in enumerate(batch(train_imgs, config.batchSize)):
            ori_imgs = np.array([i[1] for i in b]).astype(np.float32)
            ori_imgs = torch.from_numpy(ori_imgs)
            if (len(ori_imgs.shape) == 3):
                ori_imgs = torch.unsqueeze(ori_imgs, dim=1)

            if (config.DEVICE == "cuda"):
                ori_imgs = ori_imgs.cuda()

            # training bias field sub-network
            fake_bias = gen_C(ori_imgs)
            fake_prob = clu_C(ori_imgs)
            prob = fake_prob.detach()
            loss_bias = BIAS(ori_imgs, prob, fake_bias, p=2)
            loss_tv = TV(fake_bias)
            lamda = 10 #loss_bias.detach() / loss_tv.detach()
            loss_B = loss_bias + lamda * loss_tv
            sum_loss_bias = sum_loss_bias + loss_bias.item()
            opt_gen.zero_grad()
            loss_B.backward()
            opt_gen.step()

            #training clustering sub-network
            fake_bias = gen_C(ori_imgs)
            fake_prob = clu_C(ori_imgs)
            bias = fake_bias.detach()

            loss_clu = CLUS(ori_imgs, fake_prob, bias, p=2)
            loss_reg = REG(ori_imgs, fake_prob, bias, p=2)

            C_loss = loss_clu + loss_reg
            sum_loss_clus = sum_loss_clus + loss_clu.item()
            opt_clu.zero_grad()
            C_loss.backward()
            opt_clu.step()

        lr_gen.step()
        lr_clu.step()

        train_loss_bias.append(sum_loss_bias / train_num)
        train_loss_clus.append(sum_loss_clus / train_num)
        epochs_x.append(epoch)

        print('Epoch {} finished.'.format((epoch + 1)))
        print('Learning rate of ClusNet is {}, BiasNet is {}'.format(
            opt_clu.state_dict()['param_groups'][0]['lr'], opt_gen.state_dict()['param_groups'][0]['lr']))
        print('Bias loss:{}, Clustering loss:{}, Intensity loss:{}'.format(sum_loss_bias / train_num, sum_loss_clus / train_num, sum_loss_inte / train_num))
        if (min_clus_loss > (sum_loss_clus / train_num)):
            min_clus_loss = sum_loss_clus / train_num
            save_checkpoint(gen_C, opt_gen, filename=config.checkpointClusGenPath)
            save_checkpoint(clu_C, opt_clu, filename=config.checkpointClusCluPath)
            print('Checkpoint{} with the least clustering loss value is saved'.format(epoch + 1))

        if (min_bias_loss > (sum_loss_bias / train_num)):
            min_bias_loss = sum_loss_bias / train_num
            save_checkpoint(gen_C, opt_gen, filename=config.checkpointBiasGenPath)
            save_checkpoint(clu_C, opt_clu, filename=config.checkpointBiasCluPath)
            print('Checkpoint{} with the least bias loss value is saved'.format(epoch + 1))
        if config.model_save:
            save_checkpoint(gen_C, opt_gen, filename=config.checkpointGen)
            save_checkpoint(clu_C, opt_clu, filename=config.checkpointClu)
    train_lossT = [train_loss_bias, train_loss_clus]
    train_lossT = np.array(train_lossT)
    train_loss = train_lossT.T
    data_loss = pd.DataFrame(train_loss)
    data_loss.columns = ['BiasLoss', 'ClusterLoss']
    # writexcel = pd.ExcelWriter(config.dir_excel)
    # data_loss.to_excel(writexcel, 'page_1', float_format='%.15f')
    # writexcel.save()

    line_clus, = plt.plot(epochs_x, train_loss_clus, color='red', linewidth=1.0, linestyle='--')
    line_bias, = plt.plot(epochs_x, train_loss_bias, color='green', linewidth=1.0, linestyle=':')
    plt.title("The training curves")
    plt.legend(handles=[line_clus, line_bias], labels=['MSE loss of corr', 'MSE loss of bias'])
    plt.show()

def main():

    gen_C = Generator(in_chan=1).to(config.DEVICE)
    clu_C = ClusterNet(in_chan=1, out_chan=config.classNum).to(config.DEVICE)
    gen_C.apply(weights_init)
    clu_C.apply(weights_init)

    opt_gen = optim.Adam(
        gen_C.parameters(),
        lr=config.learningRate,
        betas=(0.9, 0.999),
    )
    opt_clu = optim.Adam(
        clu_C.parameters(),
        lr=config.learningRate,
        betas=(0.9, 0.999),
    )

    lr_gen = lr_scheduler.ExponentialLR(opt_gen, gamma=0.9999, last_epoch=-1)
    lr_clu = lr_scheduler.ExponentialLR(opt_clu, gamma=0.9999, last_epoch=-1)

    BIAS = BiasPredictLoss()
    CLUS = ClusterLoss()
    REG = RegularizerLoss()
    TV = TVLoss()

    orimg_path = config.trainDir + 'Corrupted_data/'
    ori_ids = get_ids(orimg_path, config.datasetName)

    train_fn(gen_C, clu_C, orimg_path, ori_ids,
             opt_gen, opt_clu,BIAS, CLUS, REG, TV, lr_gen,
             lr_clu)

if __name__ == "__main__":
    main()


