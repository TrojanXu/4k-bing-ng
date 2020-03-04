import yaml
import os
import cv2
import copy
import numpy as np
import sys
from wallpaper_util import set_wallpaper
from enhancement import Denoise, AddCaption, SuperResolution
from download import download_ng_image, download_bing_image
from image_library import ImageLibrary


def load_config(path):
    with open(path, mode='r') as f:
        config = yaml.load(f)
    return config


def fetch_image(config):
    if config['image_source'] == 'bing':
        raw_img = download_bing_image()
    elif config['image_source'] == 'ng':
        raw_img = download_ng_image()
    else:
        raise NotImplementedError("unrecognized image source")

    return raw_img


if __name__ == "__main__":
    config = load_config('config.yml')
    lib = ImageLibrary(config)
    img = fetch_image(config)
    enhanced_img = copy.deepcopy(img)
    steps = [SuperResolution(2)]
    for step in steps:
        step.execute(enhanced_img)

    lib.append(img.name + '.' + config['library'][config['image_source']]['suffix'], img.data, enhanced_img.data)
    
    set_wallpaper(lib.get_latest_img_path())