import torch
import os
import re
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from utils import load_checkpoint
from torch.utils.data import DataLoader
from PIL import Image
import config
import numpy as np
from pre_processed import img_Normalize, hwc_to_chw, fore_Mask
from Bias_net import Generator
from Clus_net import ClusterNet
import time
imgsuffix = '.png'
bias_name = 'bias.png'

def mask_to_image(image):
    return image * 255

def predict_img(biasnet, prednet, origin_img, gpu=True):
    biasnet.eval()
    foremask = fore_Mask(origin_img)
    img = img_Normalize(origin_img)

    transimg = hwc_to_chw(img)
    transmask = hwc_to_chw(foremask)

    transimg = torch.from_numpy(transimg).unsqueeze(0)
    transmask = torch.from_numpy(transmask).unsqueeze(0)

    if(gpu):
        transimg = transimg.cuda()
        transmask = transmask.cuda()
    with torch.no_grad():
        outbias = biasnet(transimg)

    imgcorrect = transimg / outbias * transmask

    return imgcorrect

    # with torch.no_grad():
    #     outprob = prednet(transimg)
    #
    # return outprob

# def fileSort(filenames):
#     filenum = len(filenames)
#     suffix = filenames[0][-4:]
#     for i in range(filenum):
#         numsInfile = re.findall(r'\d+',filenames[i])
#         if(int(numsInfile[1]) < 10):
#             numsInfile[1] = '00' + numsInfile[1]
#         elif(int(numsInfile[1]) < 100):
#             numsInfile[1] = '0' + numsInfile[1]
#         filenames[i] = numsInfile[0] + numsInfile[1]
#
#     filenames.sort()
#     for i in range(filenum):
#         filenameF = filenames[i][:6]
#         filenameS = str(int(filenames[i][6:]))
#         filenames[i] = filenameF + '_' + filenameS + suffix
#     return filenames
def fileSort(filename):
    filename.sort(key=lambda x:int(x[:-4]))
    return filename

def predict_fn(gen_C, prednet, root_bias):
    gen_C = gen_C.cuda()
    gen_C.eval()
    bias_images = os.listdir(root_bias)
    bias_images = fileSort(bias_images)
    filenum = len(bias_images)
    # count = 1
    start = time.time()

    for i in range(filenum):
        bias_image_name = bias_images[i]
        bias_image_path = os.path.join(root_bias, bias_image_name)
        bias_image = np.array(Image.open(bias_image_path).convert('L'),dtype=np.float32)
        imgcorrect = predict_img(gen_C, prednet, bias_image)
        imgcorrect = mask_to_image(imgcorrect)
        img = torch.squeeze(imgcorrect)
        ###########################
        img = img.cpu().numpy()
        cv2.imwrite(config.saveDir+bias_image_name,img)
        ##########################
        # for i in range(4):
        #     img1 = imgcorrect[0,i,:,:]
        #     img1 = torch.squeeze(img1)
        #     img1 = img1.cpu().numpy()
        #     img1 = img1.astype('uint8')
        #     cv2.imwrite(config.saveDir + str(count) + imgsuffix, img1)
        #     count = count + 1
    end = time.time()
    time_sum = end - start
    time_sum = time_sum / filenum
    print(time_sum)
def main():

    root_bias = config.testDir
    gen_C = Generator(in_chan=1).to(config.DEVICE)
    pred_net = ClusterNet(in_chan=1, out_chan=config.classNum).to(config.DEVICE)
    load_checkpoint(config.checkpointBiasGenPath, gen_C)
    load_checkpoint(config.checkpointBiasCluPath, pred_net)
    # load_checkpoint('/home/liang/Data/JBHI24Data/checkpoint/BrainWeb/T1PDNet_clus_bias_genClus.pth.tar', gen_C)
    # load_checkpoint('/home/liang/Data/JBHI24Data/checkpoint/BrainWeb/T1PDNet_clus_bias_cluBias.pth.tar', pred_net)
    # gen_C.load_state_dict(torch.load(config.checkpointClusGenPath))
    # pred_net.load_state_dict(torch.load(config.checkpointClusCluPath))
    predict_fn(gen_C, pred_net, root_bias)


if __name__ == "__main__":
    main()