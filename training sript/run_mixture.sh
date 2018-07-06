#!/usr/bin/env sh

../../caffe/build/tools/caffe train \
--solver=solver_mixture.prototxt \
--gpu=0
