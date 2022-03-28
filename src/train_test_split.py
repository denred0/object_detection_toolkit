import os
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from my_utils import get_all_files_in_folder


def generate_train_test(data_dir, split):
    images = get_all_files_in_folder(data_dir, ['*.jpg'])

    with open('data/train_test_split/' + split + '.txt', "w") as outfile:
        for image in images:
            outfile.write(split + '/' + image.name)
            outfile.write("\n")
        outfile.close()


# darknet_path = Path('/home/vid/hdd/projects/darknet/my_data')

root_dir = Path('data/train_test_split/dataset')
# root_data_jpg_dir = Path('data/darknet_prepare_for_train/data_jpg')
# root_data_txt_dir = Path('data/darknet_prepare_for_train/data_txt')
train_dir = Path('data/train_test_split/train')
test_dir = Path('data/train_test_split/test')
# backup_dir = Path('data/train_test_split/backup')

# dirpath = root_data_jpg_dir
# if dirpath.exists() and dirpath.is_dir():
#     shutil.rmtree(dirpath)
# Path(dirpath).mkdir(parents=True, exist_ok=True)
#
# dirpath = root_data_txt_dir
# if dirpath.exists() and dirpath.is_dir():
#     shutil.rmtree(dirpath)
# Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = train_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = test_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

# dirpath = backup_dir
# if dirpath.exists() and dirpath.is_dir():
#     shutil.rmtree(dirpath)
# Path(dirpath).mkdir(parents=True, exist_ok=True)

all_images = get_all_files_in_folder(str(root_dir), ['*.jpg'])
all_txts = get_all_files_in_folder(str(root_dir), ['*.txt'])
print(f'Total images: {len(all_images)}')
print(f'Total labels: {len(all_txts)}')

val_part = 0.2

labels = []
images_list = []
for txt in tqdm(all_txts):
    lines = loadtxt(str(txt), delimiter=' ', unpack=False)
    if lines.shape.__len__() == 1:
        lines = [lines]

    for line in lines:
        if len(line) != 0:
            labels.append(int(line[0]))
        images_list.append(txt.stem)

# classes + counts
labels_dict = pd.DataFrame(labels, columns=["x"]).groupby('x').size().to_dict()
all_labels = sum(labels_dict.values())
print('labels_dict', labels_dict)

labels_parts = []
for key, value in labels_dict.items():
    labels_parts.append(value / all_labels)

print('labels_parts', labels_parts)
print('classes ', len(labels_parts))

straify = True
min_part = 0.2
if np.min(labels_parts) < min_part:
    straify = True

type = 2  # or 1

if straify:

    if type == 1:
        # add 0.05 for accuracy stratification
        val_part += 0.05

        # collect all classes

        # stratify
        X_train, X_test, y_train, y_test = train_test_split(images_list, labels, test_size=val_part, random_state=42,
                                                            stratify=labels, shuffle=True)
        # remove dublicates
        X_train = np.unique(X_train).tolist()
        X_test = np.unique(X_test).tolist()

        # get images that exist in train and test
        dublicates = []
        for xtr in tqdm(X_train):
            for xtt in X_test:
                if xtr == xtt:
                    dublicates.append(xtr)

        # delete such images from train and test
        for dubl in dublicates:
            X_train.remove(dubl)
            X_test.remove(dubl)

        # add dubl images in train and test with stratify
        for i, dubl in tqdm(enumerate(dublicates)):
            if i % int((10 - (val_part) * 10)) == 0:
                X_test.append(dubl)
            else:
                X_train.append(dubl)

        # copy images and txts
        for name in tqdm(X_train):
            shutil.copy(root_dir.joinpath(name + '.jpg'), train_dir)
            shutil.copy(root_dir.joinpath(name + '.txt'), train_dir)

        for name in tqdm(X_test):
            shutil.copy(root_dir.joinpath(name + '.jpg'), test_dir)
            shutil.copy(root_dir.joinpath(name + '.txt'), test_dir)

        # check stratification
        all_txt_train = get_all_files_in_folder(train_dir, ['*.txt'])

        # collect train classes and compare with all classes
        labels_train = []
        for txt in tqdm(all_txt_train):
            lines = loadtxt(str(txt), delimiter=' ', unpack=False).tolist()
            if not isinstance(lines[0], list):
                lines = [lines]

            for line in lines:
                labels_train.append(line[0])

        labels_train_dict = pd.DataFrame(labels_train, columns=["x"]).groupby('x').size().to_dict()

        st = []
        for key, value in labels_dict.items():
            val = labels_train_dict[key] / value
            st.append(val)

            print(f'Class {key} | counts {value} | test_part {val}')

        print('Train part:', np.mean(st))
    else:
        labels_dict[-1] = 99999999

        # assign one class to image  - rarest class
        x_all = []
        labels_all = []
        for txt in tqdm(all_txts):
            lines = loadtxt(str(txt), delimiter=' ', unpack=False)

            if lines.shape.__len__() == 1:
                lines = [lines]

            for line in lines:
                if len(line) != 0:
                    labels.append(int(line[0]))
                images_list.append(txt.stem)

            lab = []
            for line in lines:
                if len(line) != 0:
                    lab.append(int(line[0]))

            best_cat = -1
            x_all.append(txt.stem)
            for l in lab:
                if labels_dict[l] < labels_dict[best_cat]:
                    best_cat = l
            labels_all.append(best_cat)

        # stratify
        X_train, X_test, y_train, y_test = train_test_split(x_all, labels_all, test_size=val_part, random_state=42,
                                                            shuffle=True)

        # copy images and txts
        for name in tqdm(X_train):
            shutil.copy(root_dir.joinpath(name + '.jpg'), train_dir)
            shutil.copy(root_dir.joinpath(name + '.txt'), train_dir)

        for name in tqdm(X_test):
            shutil.copy(root_dir.joinpath(name + '.jpg'), test_dir)
            shutil.copy(root_dir.joinpath(name + '.txt'), test_dir)

        # check stratification
        all_txt_train = get_all_files_in_folder(train_dir, ['*.txt'])

        # collect train classes and compare with all classes
        labels_train = []
        for txt in tqdm(all_txt_train):
            lines = loadtxt(str(txt), delimiter=' ', unpack=False)
            if lines.shape.__len__() == 1:
                lines = [lines]

            # if not isinstance(lines[0], list):
            #     lines = [lines]

            for line in lines:
                if len(line) != 0:
                    labels_train.append(line[0])

        labels_train_dict = pd.DataFrame(labels_train, columns=["x"]).groupby('x').size().to_dict()

        st = []
        labels_dict.pop(-1)
        for key, value in labels_dict.items():
            val = labels_train_dict[key] / value
            st.append(val)

            print(f'Class {key} | counts {value} | test_part {val}')

        print('Train part:', np.mean(st))

else:

    # for img in tqdm(all_images):
    #     shutil.copy(img, root_data_jpg_dir)
    #
    # for txt in tqdm(all_txts):
    #     shutil.copy(txt, root_data_txt_dir)

    np.random.shuffle(all_images)
    train_FileNames, val_FileNames = np.split(np.array(all_images), [int(len(all_images) * (1 - val_part))])

    for name in tqdm(train_FileNames):
        shutil.copy(name, train_dir)
        shutil.copy(root_dir.joinpath(name.stem + '.txt'), train_dir)

    for name in tqdm(val_FileNames):
        shutil.copy(name, test_dir)
        shutil.copy(root_dir.joinpath(name.stem + '.txt'), test_dir)

generate_train_test(train_dir, 'train')
generate_train_test(test_dir, 'test')

# copy cfg data
# shutil.copy('data/darknet_prepare_for_train/0_cfg/obj.data', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_cfg/obj.names', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_cfg/yolov4-obj-mycustom.cfg', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_weights/yolov4-p5.conv.232', darknet_path)

# os.system("/home/vid/hdd/projects/darknet/darknet detector train "
#           "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_cfg/obj.data "
#           "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_cfg/yolov4-obj-mycustom.cfg "
#           "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_weights/yolov4-p5.conv.232 -map")

# ./darknet detector train my_data/obj.data my_data/yolov4-obj-mycustom.cfg my_data/yolov4-p5.conv.232 -dont_show -map
