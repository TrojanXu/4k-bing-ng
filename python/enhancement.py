import sys
#import tensorflow as tf
sys.path.append("../3rdparty/mmsr/codes/")
import models.archs.RRDBNet_arch as arch
import utils.util as util
import numpy as np
import torch
import onnxruntime as rt
import argparse
import glob
import cv2
from image_content import ImageContent

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

class Step:
    def get_description(self):
        return "Step"
    pass

class Denoise(Step):
    def __init__(self, model_path):
        super(Denoise, self).__init__()


        self._model_path = model_path
        '''
        # tensorflow
        self._graph = tf.Graph()
        self._sess = tf.InteractiveSession(graph = self._graph)

        with tf.gfile.GFile(self._model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Define input tensor
        self._input = tf.placeholder(tf.float32, shape = [None, 3, None, None], name='Inputs/Placeholder')
        tf.import_graph_def(graph_def, {'Inputs/Placeholder': self._input})        
        self._output_tensor = self._graph.get_tensor_by_name("import/noise2clean_1/nin_c/add:0")     
        '''   
        self._tile_size = 128
        
        self._sess = rt.InferenceSession(self._model_path, providers=['CPUExecutionProvider'])
        sess_opt = self._sess.get_session_options()
        self._input = self._sess.get_inputs()[0].name

    def get_description(self):
        return "denoise"

    # assume in_img is of [0,255] and hwc
    def execute(self, in_img):
        tile_size = self._tile_size
        data_type = in_img.data.dtype

        data_nchw = in_img.get_nchw_data(tile=[tile_size, tile_size])
        data_nchw = adjust_dynamic_range(data_nchw, [0, 255], [0., 1.])
        out = self._sess.run(None, {self._input:data_nchw})[0]

        out = adjust_dynamic_range(out, [0.,1.], [0, 255])
        out = np.rint(out).clip(0, 255).astype(data_type)

        in_img.set_nchw_data(out)
        '''
        in_img_sh = in_img.data.shape
        h, w = in_img_sh[0], in_img_sh[1]
        num_tile_h, num_tile_w = (h+tile_size-1) // tile_size, (w+tile_size-1)//tile_size

        for i in range(num_tile_h):
            start_h, end_h = i*tile_size, min(i*tile_size+tile_size, h)
            for j in range(num_tile_w):
                start_w, end_w = j*tile_size, min(j*tile_size+tile_size, w)
                img = in_img.data[start_h:end_h, start_w:end_w, :]

                img = np.expand_dims(img.transpose([2,0,1]), axis=0)
                img = adjust_dynamic_range(img, [0, 255], [0.0, 1.0])
                sh = img.shape[2:]
                validation_image_size = [max([x.shape[axis] for x in [img]]) for axis in [2, 3]]
                validation_image_size = [(x + 31) // 32 * 32 for x in validation_image_size] # Round up to a multiple of 32.
                validation_image_size = [max(validation_image_size) for x in validation_image_size] # Square it up for the rotators.
                img = np.pad(img, [[0, 0], [0, 0], [0, validation_image_size[0] - sh[0]], [0, validation_image_size[1] - sh[1]]], 'reflect')
                
                out = self._sess.run(None, {self._input: img})[0]
                #out = self._sess.run(self._output_tensor, feed_dict = {self._input: img})

                out = out[0, :, :sh[0], :sh[1]].transpose([1,2,0])
                out = adjust_dynamic_range(out, [0,1], [0, 255])
                out = np.rint(out).clip(0, 255).astype(data_type)

                in_img.data[start_h:end_h, start_w:end_w, :] = out
        '''

class AddCaption(Step):
    def __init__(self):
        super(AddCaption, self).__init__()

    def execute(self, in_img):
        pass


class SuperResolution(Step):
    def __init__(self, model_path, scale):
        super(SuperResolution, self).__init__()
        assert(scale==2 or scale == 4)
        self._model_path = model_path
        self._description = 'x{}'.format(scale)

    def get_description(self):
        return self._description

    def execute(self, in_img):
        data_type = in_img.data.dtype
        img = np.transpose(in_img.data.astype(np.float32), [2, 0, 1])
        img = np.expand_dims(img, axis=0) # nchw
        img = img[:, [2, 1, 0], :, :] / 255.0
                
        if '.onnx' in self._model_path:
            pred = self._onnx_infer(img)
        else:
            pred = self._torch_infer(img)
        
        pred = util.tensor2img(pred)
        in_img.data = pred
    
    def _onnx_infer(self, img):
        sess = rt.InferenceSession(self._model_path)
        sess_opt = sess.get_session_options()
        input_name = sess.get_inputs()[0].name
        pred_onnx = sess.run(None, {input_name: img})[0]
        return torch.from_numpy(pred_onnx)
    
    def _torch_infer(self, img):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        torch_input = torch.from_numpy(img).to(device)
        model = arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23,upscale=2)
        model_bytes = torch.load(self._model_path)
        model.load_state_dict(model_bytes, strict=False)
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            pred = model(torch_input)
        return pred

'''
def enhance_a_single_image(img, steps):


    img = ImageContent(cv2.imread(img_path))
    description = ""
    for step in steps:
        step.execute(img)
        description += "_" + step.get_description()

    cv2.imwrite(img_path.replace('.png', description+'.png').replace('.jpg', description+'.jpg'), img.data)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhancement of a single image or images")
    parser.add_argument('--image-dir', help='Path to image set')
    parser.add_argument('--image', help='path to image')
    parser.add_argument('--denoise', help='denoise model')
    parser.add_argument('--x2', help='x2 model')
    parser.add_argument('--x4', help='x4 model')
    args = parser.parse_args()

    if (args.image_dir is None) == (args.image is None):
        print("both --image-dir and --image are set or unset. Please set either one.")
        exit(1)

    tasks = []
    if args.x2 is not None:
        steps = [SuperResolution(args.x2, 2)]
        if args.denoise is not None:
            steps.append(Denoise(args.denoise))
        tasks.append(steps)

    if args.x4 is not None:
        steps = [SuperResolution(args.x4, 4)]
        if args.denoise is not None:
            steps.append(Denoise(args.denoise))
        tasks.append(steps)

    if args.x2 is None and args.x4 is None and args.denoise is not None:
        tasks.append([Denoise(args.denoise)])

    if len(tasks) == 0:
        print("No model specified. Please specify at least one model.")
        exit(1)

    img_list = []
    img_path_list = []
    if args.image_dir is not None:
        img_path_list = glob.glob(args.image_dir+"/*")
    else:
        img_path_list = [arg.image]

    for task in tasks:
        img_list = []
        for img_path in img_path_list:
            if img_path.endswith(".png") or img_path.endswith(".jpg"):
                img_list.append(ImageContent(cv2.imread(img_path), img_path=img_path))        

        for step in task:
            for img in img_list:
                step.execute(img)
                img.suffix += "_" + step.get_description()
            del step

        for img in img_list:
            img.save()
