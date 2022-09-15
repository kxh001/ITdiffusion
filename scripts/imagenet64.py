import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch_train(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')
    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]

    out_dir = "/media/theo/Data/ImageNet64/Imagenet64_train/image/"
    for i in tqdm(range(data_size)):
        image = X_train[i]
        filename = os.path.join(out_dir, f"{Y_train[i]}_{i:06d}.png")
        Image.fromarray(image).save(filename)

def load_databatch_val(data_folder, img_size=32):
    data_file = os.path.join(data_folder, 'val_data')
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']

    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    X_val = x[0:data_size, :, :, :]
    Y_val = y[0:data_size]

    out_dir = "/media/theo/Data/ImageNet64/Imagenet64_val/image/"

    for i in tqdm(range(data_size)):
        image = X_val[i]
        filename = os.path.join(out_dir, f"{Y_val[i]}_{i:06d}.png")
        Image.fromarray(image).save(filename)

def main():
    data_folder_train = "/media/theo/Data/ImageNet64/Imagenet64_train"
    data_folder_val = "/media/theo/Data/ImageNet64/Imagenet64_val"

    load_databatch_train(data_folder_train, 1, 64)
    load_databatch_val(data_folder_val, 64) 


if __name__ == "__main__":
    main()
