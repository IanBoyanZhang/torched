# Coding Challenge

## Dev environment setup

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
unzip libtorch-shared-with-deps-latest.zip
```

## Build

```
mkdir build & cd build
cmake -D DOWNLOAD_DATASETS=ON -DCMAKE_PREFIX_PATH=path_to_libtorch ..
cmake --build . --config Release -v
```

## TODO: Cpp Pytorch side-by-side

[Module](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html?highlight=torch%20nn%20module#classtorch_1_1nn_1_1_module)

## References

[pytorch](https://pytorch.org/)

[Pytorch cpp minimal example and installation](https://pytorch.org/cppdocs/installing.html)

[Installing pytorch c api](https://medium.com/@albertsundjaja/installing-pytorch-c-api-d52c722f47ec)

[Pytorch cpp examples](https://github.com/prabhuomkar/pytorch-cpp)

[MNIST CPP example](https://github.com/pytorch/examples/blob/master/cpp/mnist/mnist.cpp)
