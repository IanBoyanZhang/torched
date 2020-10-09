#include "convnet.h"
#include <iostream>
#include <experimental/filesystem>
#include <torch/torch.h>

constexpr std::int64_t kNumClasses = 10;
constexpr std::int64_t kBatchSize = 100;
constexpr std::size_t kNumEpoch = 5;
constexpr double kLearningRate = 0.001;

template <typename DataLoader, typename TorchOptimizer>
std::tuple<double, double>
data_loop(LeNet5 &model, torch::Device device, DataLoader &loader,
          std::size_t num_samples, std::unique_ptr<TorchOptimizer> &optimizer) {
  /**
   * \tparam DataLoader `Pytorch DataLoader
   * <https://pytorch.org/cppdocs/api/file_torch_csrc_api_include_torch_data_dataloader.h.html#file-torch-csrc-api-include-torch-data-dataloader-h>`
   * \tparam TorchOptimizer `Pytorch Optimizer
   * <https://pytorch.org/cppdocs/api/classtorch_1_1optim_1_1_optimizer.html>`
   * \param LeNet5 implementation
   * \param device type CPU/GPU
   * \param num_samples
   * \param unique_ptr holding tparam TorchOptimizer
   * \return tuple of accuracy and mean running loss for quantifying
   * training/testing performance
   */

  double running_loss = 0.0;
  std::size_t num_correct = 0;

  double accuracy = 0.0, sample_mean_loss = 0.0;

  for (const auto &batch : *loader) {
    auto data = batch.data.to(device);
    auto target = batch.target.to(device);

    // Forward pass
    auto output = model->forward(data);

    /**
     * Calculate loss
     * I think this is a rather confusing part coming from syntax
     * cross_entropy() returns loss as a Tensor
     * `torch_nn_functional_loss.h <https://pytorch.org/cppdocs/api/program_listing_file_torch_csrc_api_include_torch_nn_functional_loss.h.html>`
     * `End to End example <https://pytorch.org/cppdocs/frontend.html>`
     * item is a template method of `Tensor <https://pytorch.org/cppdocs/api/program_listing_file_build_aten_src_ATen_core_TensorBody.h.html>`
     * For inference, compiler requires an explicit indication that '<' is `"begin template arguments" <https://stackoverflow.com/questions/3499101/when-do-we-need-a-template-construct>`
     * I guess an easier to understand syntax would be template(loss.item)<double>() or loss.item(template)<double>(), but both are illegal expressions
     *
     * References:
     * `template dot template construction usage <https://stackoverflow.com/questions/8463368/template-dot-template-construction-usage>`
     * `c template parameter inference <https://stackoverflow.com/questions/41548216/c-template-parameter-inference>`
     */
    auto loss = torch::nn::functional::cross_entropy(output, target);
    running_loss += loss.template item<double>() * data.size(0);

    // Calculate prediction
    auto prediction = output.argmax(1);

    num_correct += prediction.eq(target).sum().template item<std::int64_t>();

    if (optimizer) {
      optimizer->zero_grad();
      loss.backward();
      optimizer->step();
    }

    sample_mean_loss = running_loss / num_samples;
    accuracy = static_cast<double>(num_correct) / num_samples;
  }

  return {accuracy, sample_mean_loss};
}

template <typename DataLoader, typename TorchOptimizer>
void train(LeNet5 &model, torch::Device device, DataLoader &train_loader,
           std::unique_ptr<TorchOptimizer> &optimizer,
           std::size_t num_train_samples) {
  double accuracy, sample_mean_loss;
  for (size_t epoch = 0; epoch != kNumEpoch; ++epoch) {
    std::tie(accuracy, sample_mean_loss) =
        data_loop(model, device, train_loader, num_train_samples, optimizer);
  }
}

template <typename DataLoader, typename TorchOptim>
void test(LeNet5 &model, torch::Device device, DataLoader &test_loader,
          std::unique_ptr<TorchOptim> &optimizer,
          std::size_t num_test_samples) {
  double test_accuracy, test_sample_mean_loss;
  std::tie(test_accuracy, test_sample_mean_loss) =
      data_loop(model, device, test_loader, num_test_samples, optimizer);

  std::cout << "Testing finished!\n";
  std::cout << "Testset - Loss: " << test_sample_mean_loss
            << ", Accuracy: " << test_accuracy << '\n';
}

int main(int argc, const char *argv[]) {

  // Device
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

  std::cout << (cuda_available ? "CUDA available. Training on GPU."
                               : "Training on CPU.")
            << '\n';

  const std::string MNIST_data_path = "./data/mnist/";
  const std::string saved_model_path = "./model/model.pt";

  // MNIST dataset
  // map multi-threaded?
  // https://stackoverflow.com/questions/15067160/stdmap-thread-safety
  auto train_dataset =
      torch::data::datasets::MNIST(MNIST_data_path,
                                   torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  // Number of samples in the training set
  auto num_train_samples = train_dataset.size().value();

  auto test_dataset =
      torch::data::datasets::MNIST(MNIST_data_path,
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  // Number of samples in the test set
  auto num_test_samples = test_dataset.size().value();

  // Data loader
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(train_dataset), kBatchSize);

  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(test_dataset), kBatchSize);

  // Model
  // Net model(kNumClasses);
  LeNet5 model(kNumClasses);
  // model.to(device);
  model->to(device);

  auto optimizer = std::make_unique<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(kLearningRate));

  // Set floating point output precision
  std::cout << std::fixed << std::setprecision(4);

  if (!std::experimental::filesystem::exists(saved_model_path)) {
    // Training
    train(model, device, train_loader, optimizer,
          static_cast<std::size_t>(num_train_samples));
  } else {
    std::cout << "Loading existing model...\n";
    torch::load(model, saved_model_path);
  }

  model->eval();

  // Not being used, why?
  // torch::NoGradGuard no_grad;
  optimizer = nullptr;
  test(model, device, test_loader, optimizer,
       static_cast<std::size_t>(num_test_samples));

  // save the model
  torch::save(model, saved_model_path);

  return 0;
}
