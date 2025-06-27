from os.path import splitext
from os import listdir
from PIL import Image
from pre_processed import hwc_to_chw, img_Normalize, crop_image
import numpy as np
import random
import re

def BrainWebfileSort(filename):
    filename.sort(key=lambda x: int(x[:-4]))
    return filename

def HCPfileSort(filenames):
    filenum = len(filenames)
    suffix = filenames[0][-4:]
    for i in range(filenum):
        numsInfile = re.findall(r'\d+',filenames[i])
        if(int(numsInfile[1]) < 10):
            numsInfile[1] = '00' + numsInfile[1]
        elif(int(numsInfile[1]) < 100):
            numsInfile[1] = '0' + numsInfile[1]
        filenames[i] = numsInfile[0] + numsInfile[1]

    filenames.sort()
    for i in range(filenum):
        filenameF = filenames[i][:6]
        filenameS = str(int(filenames[i][6:]))
        filenames[i] = filenameF + '_' + filenameS + suffix
    return filenames

'''
return file names of all images in the same directory in order
'''
def get_ids(dir, dataset_name):
    filename_sort = listdir(dir)
    if(dataset_name == 'HCP'):
        filename_sort = HCPfileSort(filename_sort)
    elif(dataset_name == 'BrainWeb' or dataset_name == 'OAI'):
        filename_sort = BrainWebfileSort(filename_sort)
    return (splitext(file)[0] for file in filename_sort)

'''
return cropped images by pre-set size
'''
def to_get_image(ids,dir,suffix,height=256,width=256):
    for id in ids:
        img = Image.open(dir+id+suffix)
        gray_img = img.convert('L')
        gray_img = gray_img.resize((height,width),Image.ANTIALIAS)

        gray_img = np.array(gray_img,dtype=np.float32)
        image = gray_img
        yield image

def get_images_ids(ids,dir_img):
    imgs = to_get_image(ids,dir_img,'.png')
    img_change = map(hwc_to_chw,imgs)
    img_normalized = map(img_Normalize,img_change)

    return zip(ids,img_normalized)

def id_select(id, ori_dict, cli_ids, climg_num, slice_num = 219):
    ra_seq = 0
    seq_num = climg_num / slice_num
    id_num = ori_dict[id] #find the corresponding number of pic
    id_num = id_num % slice_num #find the corresponding number of pic whithin one seq
    if((id_num <= 4) or (id_num >= 215) or (id_num >= (climg_num - 4))):
        el_num = id_num
    else:
        ra_num = random.randint(-3,3)
        el_num = id_num + ra_num #select the adjacent pic number
    if(seq_num > 1):
        ra_seq = random.randint(0,seq_num - 1) # select which seq
    cli_num = ra_seq * slice_num + el_num #determine the final pic number
    cli_id = cli_ids[cli_num] #determine the final pic ID
    return cli_id

def id_select_BrainWeb(cli_ids, climg_num):

    cli_num = random.randint(0, climg_num - 1) #determine the final pic number
    cli_id = cli_ids[cli_num] #determine the final pic ID
    return cli_id

def to_image_aug(orimg_path, climg_path, ori_ids, cli_ids, ori_dict, suffix):
    climg_num = len(cli_ids)
    for id in ori_ids:
        image = Image.open(orimg_path+id+suffix)
        image = np.array((image.convert('L')), dtype=np.float32)

        # cli_id = id_select(id, ori_dict, cli_ids, climg_num)
        cli_id = id_select_BrainWeb(cli_ids, climg_num)
        climg = Image.open(climg_path+cli_id+suffix)
        climg = np.array((climg.convert('L')), dtype=np.float32)

        image = hwc_to_chw(image)
        image = img_Normalize(image)
        climg = hwc_to_chw(climg)
        climg = img_Normalize(climg)
        image_clear = np.concatenate((image,climg),axis=0)
        yield image_clear

def get_images_train(orimg_path, climg_path, ori_ids, cli_ids, ori_dict):
    image_mask = to_image_aug(orimg_path, climg_path, ori_ids, cli_ids, ori_dict, '.png')

    return image_mask

def to_paired_image_aug(orimg_path, climg_path, ori_ids, suffix):
    for id in ori_ids:
        image = Image.open(orimg_path+id+suffix)
        image = np.array((image.convert('L')), dtype=np.float32)

        climg = Image.open(climg_path+id+suffix)
        climg = np.array((climg.convert('L')), dtype=np.float32)

        image = hwc_to_chw(image)
        image = img_Normalize(image)
        climg = hwc_to_chw(climg)
        climg = img_Normalize(climg)
        image_clear = np.concatenate((image,climg),axis=0)
        yield image_clear

def get_paired_images_train(orimg_path, climg_path, ori_ids, cli_ids, ori_dict):
    image_mask = to_paired_image_aug(orimg_path, climg_path, ori_ids, cli_ids, ori_dict, '.png')

    return image_mask

def get_images(ids,dir_img,dir_mask):
    imgs=to_crop_image(ids,dir_img,'.png')
    img_change=map(hwc_to_chw,imgs)
    img_normalzied=map(img_Normalize,img_change)

    masks=to_crop_image(ids,dir_mask,'.png')
    mask_change = map(hwc_to_chw,masks)
    mask_normalized=map(img_Normalize,mask_change)

    return zip(img_normalzied,mask_normalized)

def to_crop_image(ids,dir,suffix):
    for id in ids:
        img = Image.open(dir+id+suffix)
        image=crop_image(img.convert('L'))
        yield image
