#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_as_prob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   // force the row sum to be 1
  Dtype * mweight = this->blobs_[0]->mutable_cpu_data();
    for(int i = 0; i < K_; ++i){
      Dtype row_sum = 0.0;
      for(int j = 0; j < N_; ++j){
        if(mweight[i*N_+j] < 0)
          mweight[i*N_+j] = minvalue_;
        row_sum += mweight[i*N_ + j];
      }
      if(row_sum >= 1e-10){
        for(int j = 0; j < N_; ++ j)
        {
          mweight[i*N_ + j] /= row_sum;
        }
      }
    }

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    if (M_ == 1) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    }else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    } 

}

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductAsProbLayer);

}  // namespace caffe
