import numpy as np
import copy
import cv2

class ImageContent(object):
    def __init__(self, img, name="", caption="", description="", credit="", img_path=""):
        self._caption = caption
        self._name = name
        self._data = img
        self._description = description
        self._credit = credit
        self._img_path=img_path
        self._nchw_data = None
        self._suffix = ""
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
    
    @property
    def caption(self):
        return self._caption

    @property
    def name(self):
        return self._name

    @property
    def credit(self):
        return self._credit

    @property
    def description(self):
        return self._description
    
    @property
    def suffix(self):
        return self._suffix
    
    @suffix.setter
    def suffix(self, value):
        self._suffix = value

    def get_nchw_data(self, tile=[512, 512], padding='reflect'):
        if self._nchw_data is not None and self._nchw_data.shape[2]==tile[0] and self._nchw_data.shape[3]==tile[1]:
          return self._nchw_data

        # self._data is in HWC
        # padding
        data = copy.deepcopy(self._data)
        h, w, c = data.shape
        x, y = tile
        if h <= x and w <= y:
            self._nchw_data = np.expand_dims(np.transpose(data, [2,0,1]), axis=0)
            self._padded_h = h
            self._padded_w = w
            return self._nchw_data
        else:
            padded_h = ((h+x-1)//x) * x
            padded_w = ((w+y-1)//y) * y
            #H'W'C
            data = np.pad(data, [[0, padded_h-h], [0, padded_w-w], [0,0]], padding)
            data = np.reshape(data, (padded_h//x, x, padded_w//y, y, c))
            data = np.transpose(data, [0,2,4,1,3])
            data = np.reshape(data, (padded_h//x * padded_w//y, c, x, y))
            self._nchw_data = data
            self._padded_h = padded_h
            self._padded_w = padded_w
            return self._nchw_data

    def _update_data_from_nchw_data(self,):
        if self._nchw_data is None:
            return
        h,w = self._data.shape[0:2]
        if self._padded_h == h and self._padded_w == w:
            self._data = np.transpose(np.squeeze(self._nchw_data, axis=0), [1,2,0])
            return
        data = copy.deepcopy(self._nchw_data)
        n, c, x, y = data.shape
        data = np.reshape(data, (self._padded_h//x, self._padded_w//y, c, x, y))
        data = np.transpose(data, [0,3,1,4,2])
        data = np.reshape(data, (self._padded_h, self._padded_w,c))
        
        self._data = data[0:h, 0:w, :]

    def set_nchw_data(self, nchw_data):
        self._nchw_data = nchw_data
        self._update_data_from_nchw_data()

    def save(self):
        if self._img_path == "":
            print("img_path is not set.")
            return
        cv2.imwrite(self._img_path.replace('.png', self._suffix+'.png').replace('.jpg', self._suffix+'.jpg'), self._data)

def test():
    img = ImageContent(np.random.rand(33,33,3))
    img_copy = copy.deepcopy(img)
    nchw_img = img.get_nchw_data(tile=[32, 32])
    img.set_nchw_data(nchw_img)
    assert((img.data == img_copy.data).all())

    img = ImageContent(np.random.rand(31,33,3))
    img_copy = copy.deepcopy(img)
    nchw_img = img.get_nchw_data(tile=[32, 32])
    img.set_nchw_data(nchw_img)    
    assert((img.data == img_copy.data).all())    

    img = ImageContent(np.random.rand(33,31,3))
    img_copy = copy.deepcopy(img)
    nchw_img = img.get_nchw_data(tile=[32, 32])
    img.set_nchw_data(nchw_img) 
    assert((img.data == img_copy.data).all())    

    img = ImageContent(np.random.rand(31,31,3))
    img_copy = copy.deepcopy(img)
    nchw_img = img.get_nchw_data(tile=[32, 32])
    img.set_nchw_data(nchw_img)
    assert((img.data == img_copy.data).all())    

    img = ImageContent(np.random.rand(65,65,1))
    img_copy = copy.deepcopy(img)
    nchw_img = img.get_nchw_data(tile=[32, 32])
    img.set_nchw_data(nchw_img)    
    assert((img.data == img_copy.data).all())    

    print("PASS")
    

if __name__ == "__main__":
    test()