from __future__ import division
import numpy as np
import sys
caffe_root = '../../../' 
sys.path.insert(0, caffe_root + 'python')
import caffe

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
#			raise Exception()
        if h != w:
            print 'filters need to be square'
            raise
#			raise Exception()
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
#base_weights = '5stage-vgg.caffemodel'
base_weights = '../../pretrainedmodel/5stage-vgg.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(3)

# solver = caffe.SGDSolver('solverBL.prototxt')
# solver = caffe.SGDSolver('solverBL1.prototxt')
solver = caffe.SGDSolver('solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
#solver.restore('dsn-full-res-3-scales_iter_29000.solverstate')
#solver.net.copy_from(base_weights)
#solver.restore('models1/_iter_160000.solverstate')
#solver.restore('/home/xbw/CNNmodels/catmore/_iter_88000.solverstate')
solver.restore('/home/xbw/CNNmodels/catmore/_iter_168000.solverstate')

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(60000)
# solver.step(100)

