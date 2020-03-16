import cv2
import os
import sys

class ImageLibrary(object):
    def __init__(self, config):
        self._image_source = config['image_source']
        self._lib_path = self._get_lib_path(config['library'][self._image_source])
        
        if not os.path.exists(self._lib_path):
            os.makedirs(self._lib_path)
            os.makedirs(os.path.join(self._lib_path,'raw'))
            os.makedirs(os.path.join(self._lib_path,'enhanced'))
        self._img_list_path = os.path.join(self._lib_path,'img_list.txt')
        self._max_images = config['library'][self._image_source]['pretain_last_images']
        self._img_list = []
        if os.path.exists(self._img_list_path):
            with open(self._img_list_path, 'r') as f:
                tmp = f.read()
                print(tmp.splitlines())
                self._img_list = tmp.splitlines()
        
    def append(self, img_name, raw, enhanced):
        if img_name in self._img_list:
            return
        if len(self._img_list) >= self._max_images:
            self._pop() 
        self._img_list.append(img_name)
        with open(self._img_list_path, 'w+') as f:
            for img_name in self._img_list:
                f.write(img_name+'\n')
        raw_img_path = os.path.join(self._lib_path, 'raw', img_name)
        enhanced_img_path = raw_img_path.replace('raw', 'enhanced')
        
        cv2.imwrite(raw_img_path, raw)
        cv2.imwrite(enhanced_img_path, enhanced)

    def get_latest_img_path(self):
        if len(self._img_list) == 0:
            return ""
        return os.path.join(self._lib_path, "enhanced", self._img_list[-1])

    def _pop(self):
        img_name = self._img_list.pop(0)
        raw_img_path = os.path.join(self._lib_path, 'raw', img_name)
        enhanced_img_path = raw_img_path.replace('raw', 'enhanced')
        os.remove(raw_img_path)
        os.remove(enhanced_img_path)

    def _get_lib_path(self, path_dict):
        if sys.platform.startswith('win32'):
            return path_dict['win_path']
        elif sys.platform.startswith('linux'):
            return path_dict['linux_path']
        else:
            return path_dict['mac_path']