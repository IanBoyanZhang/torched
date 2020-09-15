#include <torch/torch.h>
#include <iostream>
#include "convnet.h"
#include <sys/stat.h>

constexpr const int64_t kNumClasses   = 10;
constexpr const int64_t kBatchSize    = 100;
constexpr const size_t  kNumEpoch     = 5;
constexpr const double  kLearningRate = 0.001;

template <typename DataLoader>
void train(
  ConvNet& model,
  torch::Device device,
  DataLoader& train_loader,
  torch::optim::Optimizer& optimizer,
  size_t num_train_samples
  ) {
  for (size_t epoch = 0; epoch != kNumEpoch; ++epoch) {
    double running_loss = 0.0;
    size_t num_correct = 0;

    for (auto& batch: *train_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      // Forward pass
      auto output = model->forward(data);

      // Calculate loss
      auto loss = torch::nn::functional::cross_entropy(output, target);

      // https://github.com/pytorch/pytorch/issues/20287
      AT_ASSERT(!std::isnan(loss.template item<float>()));

      // Update running loss
      // running_loss += loss.item<double>() * data.size(0);
      running_loss += loss.template item<double>() * data.size(0);

      // Calculate prediction
      auto prediction = output.argmax(1);

      // Update number of correctly classified samples
      // num_correct += prediction.eq(target).sum().item<int64_t>();
      num_correct += prediction.eq(target).sum().template item<int64_t>();

      // Backward pass and optimize
      // https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/8
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }

    auto sample_mean_loss = running_loss / num_train_samples;
    auto accuracy = static_cast<double>(num_correct) / num_train_samples;

  }
}

template <typename DataLoader>
void test(
  ConvNet& model,
  torch::Device device,
  DataLoader& test_loader,
  size_t num_test_samples
  ) {
  double running_loss = 0.0;
  size_t num_correct = 0;

  // Test
  for (const auto& batch : *test_loader) {
    auto data = batch.data.to(device);
    auto target = batch.target.to(device);

    auto output = model->forward(data);

    auto loss = torch::nn::functional::cross_entropy(output, target);
    running_loss += loss.template item<double>() * data.size(0);

    auto prediction = output.argmax(1);
    num_correct += prediction.eq(target).sum().template item<int64_t>();
  }

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss = running_loss / num_test_samples;

  std::cout << "Testing finished!\n";
  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

inline bool file_exists(const std::string& name) {
  struct stat buffer;	
  return (stat (name.c_str(), &buffer) == 0);
}

int main(int argc, const char* argv[]) {

  // Device
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

  const std::string MNIST_data_path = "./data/mnist/";
  const std::string saved_model_path = "./model/model.pt";

  // MNIST dataset
  auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTrain)
	  .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	  .map(torch::data::transforms::Stack<>());

  // Number of samples in the training set
  auto num_train_samples = train_dataset.size().value();

  auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
	  .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	  .map(torch::data::transforms::Stack<>());

  // Number of samples in the test set
  auto num_test_samples = test_dataset.size().value(); 

  // Data loader
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), kBatchSize);

  auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), kBatchSize);

  // Model
  // Net model(kNumClasses);
  ConvNet model(kNumClasses);
  // model.to(device);
  model->to(device);

  // Optimizer
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(kLearningRate));

  // Set floating point output precision
  std::cout << std::fixed << std::setprecision(4);

  if (!file_exists(saved_model_path)) {
    // Training
    train(model, device, train_loader, optimizer, static_cast<size_t>(num_train_samples));
  } else {
    std::cout << "Loading existing model...\n";
    torch::load(model, saved_model_path);
  }

  model->eval();

  // Not being used, why?
  // torch::NoGradGuard no_grad;
  test(model, device, test_loader, static_cast<size_t>(num_test_samples));
  
  // save the model
  torch::save(model, saved_model_path);

  return 0; 
}
