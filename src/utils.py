import random
import os
import shutil
import numpy as np
import torch
import cv2
from tqdm import tqdm

from typing import List
from pathlib import Path


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def recreate_folders(root_dir: Path, folders_list: List) -> None:
    for directory in folders_list:
        output_dir = root_dir.joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def recreate_one_folder(root_dir: str) -> None:
    output_dir = Path(root_dir)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def BrightnessAndContrastAuto(im_source, clipHistPercent=0):
    histSize = 256
    minGray = 0
    maxGray = 0

    imgray = im_source.copy()
    if len(im_source.shape) == 3:
        imgray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)

    if clipHistPercent == 0:
        # keep full available range
        minGray, maxGray, min_pt, max_pt = cv2.minMaxLoc(imgray)
    else:
        histRange = [0, 256]
        uniform = True
        accumulate = False
        hist = cv2.calcHist([imgray], [0], None, [histSize], histRange, uniform, accumulate)
        print()
        accumulator = [0] * len(hist)
        accumulator[0] = hist[0]
        for i in range(1, histSize):
            accumulator[i] = accumulator[i - 1] + hist[i]

        # locate points that cuts at required value
        max = accumulator[-1]
        clipHistPercent *= (max / 100.0)
        clipHistPercent /= 2.0
        minGray = 0

        #  locate left cut
        while accumulator[minGray] < clipHistPercent:
            minGray += 1

        # locate right cut
        maxGray = histSize - 1
        while accumulator[maxGray] >= (max - clipHistPercent):
            maxGray -= 1

    # current range
    inputRange = maxGray - minGray
    alpha = (histSize - 1) / inputRange
    beta = -minGray * alpha

    dst = imgray.copy()
    cv2.convertScaleAbs(imgray, dst, alpha, beta)

    return dst


def improve_brightness():
    output_foldet = "data/improve_brightness/output"
    recreate_one_folder(output_foldet)

    images = get_all_files_in_folder(Path("data/improve_brightness/input"), ["*.jpg"])

    for im in tqdm(images):
        img = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
        img_edit = BrightnessAndContrastAuto(img)
        cv2.imwrite(os.path.join(output_foldet, im.name), img_edit)


def improve_brightness2():
    output_foldet = "data/improve_brightness/output"
    recreate_one_folder(output_foldet)

    images = get_all_files_in_folder(Path("data/improve_brightness/input"), ["*.jpg"])

    for im in tqdm(images):
        img = cv2.imread(str(im), cv2.IMREAD_COLOR)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        img_edit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

        cv2.imwrite(os.path.join(output_foldet, im.name), img_edit)


if __name__ == "__main__":
    improve_brightness2()
