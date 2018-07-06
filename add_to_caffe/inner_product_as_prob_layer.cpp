#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_as_prob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_as_prob_param().num_output();
  minvalue_ = this->layer_param_.inner_product_as_prob_param().minvalue();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_as_prob_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_as_prob_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // parameter initialization

    // initialize the blob as w_ii ->1 and the row sum equals 1
    // k_ x N_ 
    Dtype * weight = this->blobs_[0]->mutable_cpu_data();
    for(int i = 0; i < K_; ++i){
      Dtype row_sum = 0.0;
      for(int j = 0; j < N_; ++j){
        if(weight[i*N_+j] < 0)
          weight[i*N_+j] = minvalue_;
        if(i == j)
          weight[i*N_+j] = 999;
        row_sum += weight[i*N_ + j];
      }
      if(row_sum >= 1e-10){
        for(int j = 0; j < N_; ++ j)
        {
          weight[i*N_ + j] /= row_sum;
        }
      }
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_as_prob_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
}

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
}

template <typename Dtype>
void InnerProductAsProbLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductAsProbLayer);
#endif

INSTANTIATE_CLASS(InnerProductAsProbLayer);
REGISTER_LAYER_CLASS(InnerProductAsProb);

}  // namespace caffe
