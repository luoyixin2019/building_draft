import numpy as np
import cv2 as cv
import scipy.misc
import scipy.io
import sys
#from PIL import Image
import os
import argparse

caffe_root = '../../../'
#caffe_root = '../caffe-fcn-master/'
sys.path.insert(0, caffe_root + 'python')

import caffe

def get_predict(imin, net, height, width):
	h_limit = imin.shape[0]
	w_limit = imin.shape[1]
	im_predict = np.zeros((h_limit, w_limit, 1), dtype=np.float32)
#		p = 0
#		for y in range(0, in_.shape[1]-100, 500):
#			for x in range(0, in_.shape[2]-100, 500):
            	# patches
#				o_patch = in_[:,y:y + 600, x:x + 600]
	for y in range(0, h_limit, height):
		for x in range(0, w_limit, width):
            	# patches
			o_patch = imin[y:y + height, x:x + width,:]
#				net.blobs['data'].reshape(1, *o_patch.shape)
#				net.blobs['data'].data[...] = o_patch
#				net.blobs['input_sat'].reshape(1, *o_patch.shape)
			o_patch = o_patch.swapaxes(0, 2).swapaxes(1, 2)
			o_patch = np.asarray(o_patch, dtype=np.float32)
			net.blobs['data'].data[:,:,:,:] = o_patch
			predict = net.forward().values()[0]
			t = predict
			print(t.shape)
			t = predict[0][:,:,:]
			t = t.swapaxes(0,2).swapaxes(0,1)
#				im_predict[y:y + 500, x:x + 500,:] = t[50:550,50:550,:]
			im_predict[y:y + height, x:x + width,:] = t
	return im_predict
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', '-d', type=str)
	parser.add_argument('--img_list', '-i', type=str)
	parser.add_argument('--model', '-m', type=str)	
	parser.add_argument('--weight', '-w', type=str)
	parser.add_argument('--iter_size', '-s', type=str)
	parser.add_argument('--out_dir', '-o', type=str)
	args = parser.parse_args()
	print args

	data_root = args.data_dir
	imgl = args.img_list
	with open(data_root+imgl) as f:
#data_root = '../../data/patchtest/'
#with open(data_root+'patchtest.lst') as f:
		test_lst = f.readlines()

	test_lst = [data_root+x.strip() for x in test_lst]

	im_lst = []
	
#	caffe.set_mode_gpu()
#	caffe.set_device(0)
	caffe.set_mode_cpu()
	model_use = args.model
	model_root = args.weight
	model_size = args.iter_size
	net = caffe.Net(model_use, model_root+'_iter_'+model_size+'.caffemodel', caffe.TEST)
	result_path = args.out_dir

	for i in range(len(test_lst)):
#	im = Image.open(test_lst[i])
		im = cv.imread(test_lst[i])
#	cv.imwrite('results/r2000/'+'im45_oread.png',im)
#		im = cv.copyMakeBorder(im, 50, 50, 50, 50, cv.BORDER_REFLECT_101)
#	cv.imwrite('results/r2000/'+'im45_pad.png',im)
#		in_ = np.array(im, dtype=np.float32)
		in_ = im.astype(float)
		in_ = in_ * 0.00390625
#		pred_img = get_predict(in_,net,500,500)
		pred_img = get_predict(in_,net,750,750)
#		pred_img = get_predict(in_,net,300,300)
		im_name = os.path.basename(test_lst[i])
#		cv.imwrite(result_path+im_name[:-4]+'_pad_'+model_size+'.png',im_predict*255)
		cv.imwrite(result_path+im_name[:-4]+'_pad_'+model_size+'.png',pred_img*255)
		np.save(result_path+im_name[:-4]+'_pad_'+model_size,pred_img)
#	cv.imwrite('results/r2000/'+'im45_read.png',in_)
#	in_ = im
#		in_ = in_[:,:,::-1]
#		in_ -= np.array((104.00698793,116.66876762,122.67891434)) 
#		im_lst.append(in_)
	

#	for idx in range(0, len(im_lst)):
#		in_ = im_lst[idx]
#		in_ = in_.transpose((2,0,1))
#	cv.imwrite('results/r2000/'+'im45_in.png',in_)
#	caffe.set_mode_gpu()
#	caffe.set_device(0)
#	model_root = 'models_2000/'
#	net = caffe.Net(model_use, model_root+'_iter_5000.caffemodel', caffe.TEST)
#		o_patches = []
#		im_predict = np.zeros((in_.shape[1]-100, in_.shape[2]-100, 1), dtype=np.float32)

#			o_patch = o_patch.transpose((1,2,0))
#			cv.imwrite('results/r2000/patch_'+str(p)+'_5000.png',t*255)
#			p+=1

