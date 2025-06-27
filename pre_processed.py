import numpy as np
import random
def hwc_to_chw(img):
    if img.ndim == 2:
        img = np.expand_dims(img,axis=0)
    return img

def batch(iterable,batch_size):
    b=[]
    for i,t in enumerate(iterable):
        b.append(t)
        if (i+1)%batch_size==0:
            yield b
            b=[]
    if len(b)>1:
        yield b

def split_dataset(dataset):
    dataset=list(dataset)
    number = len(dataset)
    random.shuffle(dataset)
    return {'train':dataset[:number]}

def img_Normalize(x):
    return x/255.0

def fore_Mask(x):
    mask = np.zeros_like(x)
    H,W = mask.shape
    for i in range(H):
        for j in range(W):
            if(x[i,j] > 10):
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    return mask

def crop_image(pilimg,set_height = 256,set_width = 256):
    #pilimg = pilimg.resize((int(pilimg.size[0] / 2),int(pilimg.size[1] / 2)),Image.ANTIALIAS)#, start_h = 128, end_h = 384, start_w = 76, end_w = 332
    height = pilimg.size[0]
    width = pilimg.size[1]

    centerx = width // 2
    centery = height // 2
    left_top_x = centerx - set_width // 2
    left_top_y = centery - set_height // 2

    right_bottom_x = centerx + set_width // 2
    right_bottom_y = centery + set_height // 2

    pilimg = pilimg.crop((left_top_x, left_top_y, right_bottom_x, right_bottom_y))

    return np.array(pilimg,dtype=np.int)
