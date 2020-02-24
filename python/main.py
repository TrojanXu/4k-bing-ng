import yaml
import os
import cv2
import datetime
from urllib.request import urlopen, urlretrieve
from xml.dom import minidom
import copy

class DownloadedImage(object):
    def __init__(self, img, name, caption):
        self._caption = caption
        self._name = name
        self._data = img
    
    def get_data(self):
        return self._data
    
    def get_caption(self):
        return self._caption

    def get_name(self):
        return self._name

class Step:
    pass

class Denoise(Step):
    def __init__(self, preserved):
        super(Denoise, self).__init__()
    
    def execute(self, in_img):
        pass

class AddCaption(Step):
    def __init__(self, preserved):
        super(AddCaption, self).__init__()
    
    def execute(self, in_img):
        pass

class SuperResolution(Step):
    def __init__(self, scale):
        super(SuperResolution, self).__init__()
        self._scale = scale
    
    def execute(self, in_img):
        pass

class ImageLibrary(object):
    def __init__(self, config):
        self._image_source = config['image_source']
        self._lib_path = config['library'][self._image_source]['path']
        
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

    def _pop(self):
        img_name = self._img_list.pop(0)
        raw_img_path = os.path.join(self._lib_path, 'raw', img_name)
        enhanced_img_path = raw_img_path.replace('raw', 'enhanced')
        os.remove(raw_img_path)
        os.remove(enhanced_img_path)

def load_config(path):
    with open(path, mode='r') as f:
        config = yaml.load(f)
    return config

def download_bing_image(idx=0):
    # Getting the XML File
    try:
        usock = urlopen(''.join(['http://www.bing.com/HPImageArchive.aspx?format=xml&idx=',
                                 str(idx), '&n=1&mkt=ru-RU']))  # ru-RU, because they always have 1920x1200 resolution
    except Exception as e:
        print('Error while downloading #', idx, e)
        return
    try:
        xmldoc = minidom.parse(usock)
    # This is raised when there is trouble finding the image url.
    except Exception as e:
        print('Error while processing XML index #', idx, e)
        return
    # Parsing the XML File

    element = xmldoc.getElementsByTagName('copyright')[0]
    caption = element.firstChild.nodeValue

    # url tag
    element = xmldoc.getElementsByTagName('url')[0]
    url = 'http://www.bing.com' + element.firstChild.nodeValue
    # Get Current Date as fileName for the downloaded Picture
    now = datetime.datetime.now()
    date = now - datetime.timedelta(days=int(idx))
    name = date.strftime('%d-%m-%Y')
    print('Downloading: ', name, 'index #', idx)

    # Download and Save the Picture
    # Get a higher resolution by replacing the file name
    filename, _ = urlretrieve(url.replace('_1366x768', '_1920x1200'))
    img = cv2.imread(filename)

    return DownloadedImage(img, name, caption)

def download_ng_image():
    raise NotImplementedError("unrecognized image source")
    pass
   
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
    steps = [Denoise(0), SuperResolution(2), AddCaption(0)]
    for step in steps:
        step.execute(enhanced_img)

    lib.append(img.get_name() + '.' + config['library'][config['image_source']]['suffix'], img.get_data(), enhanced_img.get_data())
