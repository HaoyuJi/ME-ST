import argparse
import glob
import os
from typing import List
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="convert np.array to images.")
    parser.add_argument(
        "dir",
        type=str,
        help="path to a directory containing arrays you want to convert",
    )

    return parser.parse_args()


def convert_arr2img(arr: np.ndarray) -> Image.Image:
    """
    Args:
        arr: 1d array(T, )
        palette: color palette
    """
    voc = Image.open("/chenbw/Projects/RobuSeg/imgs/voc_sample.png")
    voc = voc.convert("P")
    palette = voc.getpalette()
    arr = arr.astype(np.uint8)
    arr = np.tile(arr, (100, 1))
    img = Image.fromarray(arr)
    img = img.convert("P")
    img.putpalette(palette)
    return img

def multi_arr_to_img(multi_arr: np.ndarray,name,path='/home/chenbw/Vis/',name_list=None) -> Image.Image:
    """
    Args:
        arr: 1d array(T, )
        palette: color palette
    """
    voc = Image.open("/home/chenbw/palette.png")
    voc = voc.convert("P")
    palette = voc.getpalette()
    nums = len(multi_arr)
    plt.clf()
    fig = plt.figure(figsize=(15, 15))
    rows = nums
    cols = 1
    if name_list == None:
        name_list = ['']*len(multi_arr)
    for i, (arr,_name) in enumerate(zip(multi_arr,name_list)):
        arr = arr.astype(np.uint8)
        arr = np.tile(arr, (300, 1))
        img = Image.fromarray(arr)
        img = img.convert("P")
        img.putpalette(palette)
        arr_after = np.array(img)
        ax = fig.add_subplot(rows, 1, i + 1)
        ax.set_title(name+'_'+_name, dict(fontsize=10))
        plt.imshow(arr_after)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path + name + '.png')
    plt.clf()
    plt.close()


def vis_bar(to_be_shown, base_name, path, epoch, name_list = None):
    for idx, t in enumerate(to_be_shown):
        if type(t) == torch.Tensor:
            t = t.detach().cpu().numpy()
            to_be_shown[idx] = t
    if not os.path.exists(path):
        os.makedirs(path)
    multi_arr_to_img(np.vstack(to_be_shown), base_name + '_' + str(epoch),path = path,name_list=name_list)


if __name__ == "__main__":
    multi_arr_to_img(np.random.randn(3,2000),'test')

