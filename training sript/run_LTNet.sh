#!/usr/bin/env sh

../../caffe/build/tools/caffe train \
--solver=solver_LTNet.prototxt \
--weights=snapshot/cifar10_mix_60_70_80_A_iter_50000.caffemodel \
--gpu=0
