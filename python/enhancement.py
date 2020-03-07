import sys
sys.path.append("../3rdparty/mmsr/codes/")
import models.archs.RRDBNet_arch as arch
import utils.util as util
import numpy as np
import torch
import onnxruntime as rt

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
    def __init__(self, config):
        super(SuperResolution, self).__init__()
        self._model_path = config['model_file']['X2']
    
    def execute(self, in_img):
        data_type = in_img.data.dtype
        img = np.transpose(in_img.data.astype(np.float32), [2, 0, 1])
        img = np.expand_dims(img, axis=0) # nchw
        img = img[:, [2, 1, 0], :, :] / 255.0
                
        '''
        # select onnx if you would like to. 
        # TensorRT is not implemented now, but you can run it directy with onnx model.
        sess = rt.InferenceSession("../models/exported.onnx", providers=['CUDAExecutionProvider'])
        sess.disable_fallback()
        sess_opt = sess.get_session_options()
        input_name = sess.get_inputs()[0].name
        print(in_img.data.shape)
        pred_onnx = sess.run(None, {input_name: img})[0]
        pred_onnx = torch.from_numpy(pred_onnx)
        '''
        torch_input = torch.from_numpy(img).cuda()
        device = torch.device('cuda')
        model = arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
        model.load_state_dict(torch.load(self._model_path), strict=True)
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            pred_onnx = model(torch_input)
        
        pred_onnx = util.tensor2img(pred_onnx)
        in_img.data = pred_onnx
