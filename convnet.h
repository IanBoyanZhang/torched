// Copyright 2020-present pytorch-cpp Authors
#ifndef CONVNET_HPP
#define CONVNET_HPP

#include <torch/torch.h>

class LeNet5Impl : public torch::nn::Module {
public:
  explicit LeNet5Impl(std::int64_t num_classes = 10);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Sequential layer1{
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
      torch::nn::BatchNorm2d(16), torch::nn::ReLU(),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

  torch::nn::Sequential layer2{
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
      torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

  torch::nn::Linear fc;
};

TORCH_MODULE(LeNet5);
#endif
