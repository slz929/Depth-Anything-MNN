#-- coding:utf8 --
import argparse
import time
import MNN
import MNN.numpy as np
import MNN.cv as cv2
import cv2 as opencv2
import MNN.expr as expr
import numpy

def inference(emed, img, precision, backend, thread):
    # 0. load model
    config = {}
    config['precision'] = precision
    config['backend'] = backend
    config['numThread'] = thread
    rt = MNN.nn.create_runtime_manager((config,))
    embed = MNN.nn.load_module_from_file(emed, ['image'], ['depth'], runtime_manager=rt)
   
    # 1. preprocess
    image = cv2.imread(img)
    origin_h, origin_w, _ = image.shape
    length = 518
    if origin_h > origin_w:
        new_w = round(origin_w * float(length) / origin_h)
        new_h = length
    else:
        new_h = round(origin_h * float(length) / origin_w)
        new_w = length
    scale_w = new_w / origin_w
    sclae_h = new_h / origin_h
    input_var = cv2.resize(image, (new_w, new_h), 0., 0., cv2.INTER_CUBIC, -1, [123.675, 116.28, 103.53], [1/58.395, 1/57.12, 1/57.375])
    input_var = np.pad(input_var, [[0, length - new_h], [0, length - new_w], [0, 0]], 'constant')
    input_var = np.expand_dims(input_var, 0)
    # 2. forward
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    t1 = time.time()
    output_var = embed.forward(input_var)
    t2 = time.time()
    print('# 1. times: {} ms'.format((t2 - t1) * 1000))
    depth = MNN.expr.convert(output_var, MNN.expr.NCHW)
    # 3. post process
    depth= depth.read()
    depth = opencv2.resize(depth[0, 0], (origin_w, origin_h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(numpy.uint8)
    print('after depth ', depth.shape)
    depth_color = opencv2.applyColorMap(depth, opencv2.COLORMAP_INFERNO)
    print(depth_color.shape)

    opencv2.imwrite('res.jpg', depth_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', type=str, required=False, default= '../dam.mnn',help='the embedding model path')
    parser.add_argument('--img', type=str, required=False, default='../image.jpg', help='the input image path')
    parser.add_argument('--precision', type=str, default='normal', help='inference precision: normal, low, high, lowBF')
    parser.add_argument('--backend', type=str, default='CPU', help='inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI')
    parser.add_argument('--thread', type=int, default=4, help='inference using thread: int')
    args = parser.parse_args()
    inference(args.embed, args.img, args.precision, args.backend, args.thread)