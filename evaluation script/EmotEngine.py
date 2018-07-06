# -*- coding: utf-8 -*-
"""
Created on June 12 15:11:43 2017

@author: dualplus
"""

caffe_root = '/home/jiabei/caffe/'
import os
import sys
sys.path.insert(0, '/home/jiabei/caffe/python')
import caffe
#import lmdb
#import fileinput,re
#import math
import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
import scipy.io

sys.path.insert(0, 'src')
import random

	
class EmotEngine():
	def __init__(self, model_def, model_weights, 
			device_no = -1,
			input_data_str = 'data',
			output_str = 'output',
			norm_flag = 0,
            src_size = (256,256)):

		self.model_def = model_def
		self.model_weights = model_weights
		self.device_no = device_no
		self.input_data_str = input_data_str
		self.output_str = output_str
		self.src_size = src_size
		
		if self.device_no == -1:
			caffe.set_mode_cpu()
		else:
			caffe.set_mode_gpu()
			caffe.set_device(self.device_no)
			
		self.net = caffe.Net(self.model_def,1, weights=self.model_weights)
		self.imgshape = self.net.blobs[self.input_data_str].data.shape
                tmplist = list(self.imgshape)
                self.batchsize = tmplist[0]
                tmplist[0] = 1
                self.imgshape = tuple(tmplist)
	
		# create transformer fot e input called 'data'
		self.transformer = caffe.io.Transformer({self.input_data_str: self.net.blobs[self.input_data_str].data.shape})
		self.transformer.set_transpose(self.input_data_str,(2,0,1)) # move image channels to outermost dimension
		self.transformer.set_raw_scale(self.input_data_str,255)     # rescale from [0,1] to [0,255]
		if norm_flag == 1:
			self.transformer.set_input_scale(self.input_data_str, 0.00392156862745)
 		elif norm_flag == 2:
			self.transformer.set_input_scale(self.input_data_str, 0.0078125)
			self.transformer.set_mean(self.input_data_str, np.array([128,128,128]))
		self.transformer.set_channel_swap(self.input_data_str,[2,1,0]) # swap channels from RGB to BGR
	
	def reshape_feature(self, feat):
		feat_shape = feat.shape
		dim = 1
		for i in feat_shape:
			dim *= i
		feat = feat.reshape(dim)
		new_feat = feat.tolist()
		return new_feat
		
	def crop_center(self,img):
		crop_size = self.net.blobs[self.input_data_str].data.shape
		src_size = self.src_size
		center = np.array(src_size)/2.0
		crop_size = np.array(crop_size)[2:4]
		crop = np.tile(center, (1, 2))[0]+np.concatenate([-crop_size/2.0,crop_size/2.0])
		crop = crop.astype(int)
		img = img[crop[0]:crop[2], crop[1]:crop[3], :]
		return img
	'''
	idx = 0: neutral
	idx = 1: anger
	idx = 2: disgust
	idx = 3: fear
	idx = 4: happy
	idx = 5: sadness
	idx = 6: suprise 
	'''
	def get_emoti_from_img(self, imgpath):
		image = caffe.io.load_image(imgpath)
		image = caffe.io.resize_image(image, self.src_size)
		image = self.crop_center(image)
		transformed_img = self.transformer.preprocess(self.input_data_str, image)
		self.net.blobs[self.input_data_str].data[0] = transformed_img
		self.net.forward()
		prob = self.net.blobs[self.output_str].data
		prob = self.reshape_feature(prob)
		return prob
		
	def get_emoti_and_loss_from_img(self, imgpath, gtlbl):
		image = caffe.io.load_image(imgpath)
		image = caffe.io.resize_image(image, self.src_size)
		image = self.crop_center(image)
		transformed_img = self.transformer.preprocess(self.input_data_str, image)
		self.net.blobs[self.input_data_str].data[0] = transformed_img
		self.net.blobs["label"].data[0] = gtlbl
		self.net.forward()
		prob = self.net.blobs["output"].data
		prob = self.reshape_feature(prob)
		is_predict_correct = 0
		if prob[gtlbl] == max(prob):
			is_predict_correct = 1
		loss = self.net.blobs["loss"].data
		loss = self.reshape_feature(loss)
		return is_predict_correct, loss[0]
		
	def get_loss_from_img(self, imgpath, gtlbl):
		image = caffe.io.load_image(imgpath)
		transformed_img = self.transformer.preprocess(self.input_data_str, image)
		self.net.blobs[self.input_data_str].data[0] = transformed_img
		self.net.blobs['label'].data[0] = gtlbl
		self.net.forward()
		prob = self.net.blobs[self.output_str].data
		prob = self.reshape_feature(prob)
		return prob

	def get_emoti_and_loss_from_batch(self, imgpaths, gtlbls,output_str = 'output', nclass = 7):
		batchsize = self.batchsize
		if len(imgpaths) == 0:
			return [],[]	
		for i in range(batchsize):
			if i >= len(imgpaths):
				image = caffe.io.load_image(imgpaths[-1])
				self.net.blobs["label"].data[i] = gtlbls[-1]
			else:
				image = caffe.io.load_image(imgpaths[i])
				self.net.blobs["label"].data[i] = gtlbls[i]
			image = caffe.io.resize_image(image, self.src_size)
			image = self.crop_center(image)
			transformed_img = self.transformer.preprocess(self.input_data_str, image)
			self.net.blobs[self.input_data_str].data[i] = transformed_img
		self.net.forward()
		prob = self.net.blobs[output_str].data
		prob = self.reshape_feature(prob)
		is_predict_correct = []
		for i in range(len(imgpaths)):
			is_predict_correct.append(0)
			bidx = i*nclass
			eidx = (i+1)*nclass
			is_predict_correct[i] = 1 if prob[i*nclass+gtlbls[i]]==max(prob[bidx:eidx]) else 0
		loss = self.net.blobs["loss"].data
		loss = self.reshape_feature(loss)
		return is_predict_correct, loss[0]


def parse_img_list_item(line):
	line = line.strip('\n')
	line = line.split(' ')
	return line[0], int(line[1])
	
            
