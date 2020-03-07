# 4k-bing-ng
Super-resolution application to bing daily and NG photo-of-the-day based on [ESRGAN](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf)  
Since bing daily is of size 1920x1200. Use super-resolution GAN to generate 3840x2400 images to meet 4K Display.  
See 2020-03-07's Bing daily detail comparison.  
![avatar](LRHR_20200307.png)  
Currently I only uploaded x2 model.

# Dependency
1. PyTorch
2. Onnx runtime (optional)
3. TensorRT (optional)
4. MMSR
5. opencv-python

# How to run
1. Git clone this project.
2. git submodule update --init --recursive
3. Download latest x2 model (链接：https://pan.baidu.com/s/1eENxuBYsGTB8nTBbfaHX4w 提取码：o73z ) and put it in ./models
4. Remember to modify python/config.yml to set library path
5. python main.py to fetch today's Bing Daily
6. (Optional) Set this as a scheduled task to set it as wallpaper everyday.

It needs ~5 mins to run on CPU while 2 secs on GPU. (It's quite heavy.)