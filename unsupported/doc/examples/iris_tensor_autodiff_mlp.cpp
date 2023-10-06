#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Author: Luiz Carlos d'Oleron doleron@gmail.com, 2023
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * This is an example of using Tensors with automatic differentiation for training a simple neural network
 * to classify the Iris dataset (https://archive.ics.uci.edu/dataset/53/iris).
 *
 * The network is a regular example of MLP network having only two layers:
 * 
 * - hidden layer: a dense layer with 32 units and ReLU activation
 * - output layer: a dense layer with 3 units and Softmax activation
 *
 * The training uses Gradient Descent to minimize a categorical cross-entropy cost function.
 * 
 * The network performance is evaluated on validation data (split of 20%) by accuracy.
 *
 */

typedef typename Eigen::AutoDiffScalar<Eigen::VectorXf> AutoDiff_T;

std::random_device rd{};
const auto seed = rd();
std::mt19937 rng(seed);

/**
 * Function to load the Iris dataset.
 * To keep the example self-contained, the dataset registers were included into the source code.
 */
std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
load_iris_dataset() {
  Eigen::Tensor<float, 2> raw_data(150, 7);
  raw_data.setValues({{5.1, 3.5, 1.4, 0.2, 1, 0, 0}, {4.9, 3, 1.4, 0.2, 1, 0, 0},   {4.7, 3.2, 1.3, 0.2, 1, 0, 0},
                      {4.6, 3.1, 1.5, 0.2, 1, 0, 0}, {5, 3.6, 1.4, 0.2, 1, 0, 0},   {5.4, 3.9, 1.7, 0.4, 1, 0, 0},
                      {4.6, 3.4, 1.4, 0.3, 1, 0, 0}, {5, 3.4, 1.5, 0.2, 1, 0, 0},   {4.4, 2.9, 1.4, 0.2, 1, 0, 0},
                      {4.9, 3.1, 1.5, 0.1, 1, 0, 0}, {5.4, 3.7, 1.5, 0.2, 1, 0, 0}, {4.8, 3.4, 1.6, 0.2, 1, 0, 0},
                      {4.8, 3, 1.4, 0.1, 1, 0, 0},   {4.3, 3, 1.1, 0.1, 1, 0, 0},   {5.8, 4, 1.2, 0.2, 1, 0, 0},
                      {5.7, 4.4, 1.5, 0.4, 1, 0, 0}, {5.4, 3.9, 1.3, 0.4, 1, 0, 0}, {5.1, 3.5, 1.4, 0.3, 1, 0, 0},
                      {5.7, 3.8, 1.7, 0.3, 1, 0, 0}, {5.1, 3.8, 1.5, 0.3, 1, 0, 0}, {5.4, 3.4, 1.7, 0.2, 1, 0, 0},
                      {5.1, 3.7, 1.5, 0.4, 1, 0, 0}, {4.6, 3.6, 1, 0.2, 1, 0, 0},   {5.1, 3.3, 1.7, 0.5, 1, 0, 0},
                      {4.8, 3.4, 1.9, 0.2, 1, 0, 0}, {5, 3, 1.6, 0.2, 1, 0, 0},     {5, 3.4, 1.6, 0.4, 1, 0, 0},
                      {5.2, 3.5, 1.5, 0.2, 1, 0, 0}, {5.2, 3.4, 1.4, 0.2, 1, 0, 0}, {4.7, 3.2, 1.6, 0.2, 1, 0, 0},
                      {4.8, 3.1, 1.6, 0.2, 1, 0, 0}, {5.4, 3.4, 1.5, 0.4, 1, 0, 0}, {5.2, 4.1, 1.5, 0.1, 1, 0, 0},
                      {5.5, 4.2, 1.4, 0.2, 1, 0, 0}, {4.9, 3.1, 1.5, 0.1, 1, 0, 0}, {5, 3.2, 1.2, 0.2, 1, 0, 0},
                      {5.5, 3.5, 1.3, 0.2, 1, 0, 0}, {4.9, 3.1, 1.5, 0.1, 1, 0, 0}, {4.4, 3, 1.3, 0.2, 1, 0, 0},
                      {5.1, 3.4, 1.5, 0.2, 1, 0, 0}, {5, 3.5, 1.3, 0.3, 1, 0, 0},   {4.5, 2.3, 1.3, 0.3, 1, 0, 0},
                      {4.4, 3.2, 1.3, 0.2, 1, 0, 0}, {5, 3.5, 1.6, 0.6, 1, 0, 0},   {5.1, 3.8, 1.9, 0.4, 1, 0, 0},
                      {4.8, 3, 1.4, 0.3, 1, 0, 0},   {5.1, 3.8, 1.6, 0.2, 1, 0, 0}, {4.6, 3.2, 1.4, 0.2, 1, 0, 0},
                      {5.3, 3.7, 1.5, 0.2, 1, 0, 0}, {5, 3.3, 1.4, 0.2, 1, 0, 0},   {7, 3.2, 4.7, 1.4, 0, 1, 0},
                      {6.4, 3.2, 4.5, 1.5, 0, 1, 0}, {6.9, 3.1, 4.9, 1.5, 0, 1, 0}, {5.5, 2.3, 4, 1.3, 0, 1, 0},
                      {6.5, 2.8, 4.6, 1.5, 0, 1, 0}, {5.7, 2.8, 4.5, 1.3, 0, 1, 0}, {6.3, 3.3, 4.7, 1.6, 0, 1, 0},
                      {4.9, 2.4, 3.3, 1, 0, 1, 0},   {6.6, 2.9, 4.6, 1.3, 0, 1, 0}, {5.2, 2.7, 3.9, 1.4, 0, 1, 0},
                      {5, 2, 3.5, 1, 0, 1, 0},       {5.9, 3, 4.2, 1.5, 0, 1, 0},   {6, 2.2, 4, 1, 0, 1, 0},
                      {6.1, 2.9, 4.7, 1.4, 0, 1, 0}, {5.6, 2.9, 3.6, 1.3, 0, 1, 0}, {6.7, 3.1, 4.4, 1.4, 0, 1, 0},
                      {5.6, 3, 4.5, 1.5, 0, 1, 0},   {5.8, 2.7, 4.1, 1, 0, 1, 0},   {6.2, 2.2, 4.5, 1.5, 0, 1, 0},
                      {5.6, 2.5, 3.9, 1.1, 0, 1, 0}, {5.9, 3.2, 4.8, 1.8, 0, 1, 0}, {6.1, 2.8, 4, 1.3, 0, 1, 0},
                      {6.3, 2.5, 4.9, 1.5, 0, 1, 0}, {6.1, 2.8, 4.7, 1.2, 0, 1, 0}, {6.4, 2.9, 4.3, 1.3, 0, 1, 0},
                      {6.6, 3, 4.4, 1.4, 0, 1, 0},   {6.8, 2.8, 4.8, 1.4, 0, 1, 0}, {6.7, 3, 5, 1.7, 0, 1, 0},
                      {6, 2.9, 4.5, 1.5, 0, 1, 0},   {5.7, 2.6, 3.5, 1, 0, 1, 0},   {5.5, 2.4, 3.8, 1.1, 0, 1, 0},
                      {5.5, 2.4, 3.7, 1, 0, 1, 0},   {5.8, 2.7, 3.9, 1.2, 0, 1, 0}, {6, 2.7, 5.1, 1.6, 0, 1, 0},
                      {5.4, 3, 4.5, 1.5, 0, 1, 0},   {6, 3.4, 4.5, 1.6, 0, 1, 0},   {6.7, 3.1, 4.7, 1.5, 0, 1, 0},
                      {6.3, 2.3, 4.4, 1.3, 0, 1, 0}, {5.6, 3, 4.1, 1.3, 0, 1, 0},   {5.5, 2.5, 4, 1.3, 0, 1, 0},
                      {5.5, 2.6, 4.4, 1.2, 0, 1, 0}, {6.1, 3, 4.6, 1.4, 0, 1, 0},   {5.8, 2.6, 4, 1.2, 0, 1, 0},
                      {5, 2.3, 3.3, 1, 0, 1, 0},     {5.6, 2.7, 4.2, 1.3, 0, 1, 0}, {5.7, 3, 4.2, 1.2, 0, 1, 0},
                      {5.7, 2.9, 4.2, 1.3, 0, 1, 0}, {6.2, 2.9, 4.3, 1.3, 0, 1, 0}, {5.1, 2.5, 3, 1.1, 0, 1, 0},
                      {5.7, 2.8, 4.1, 1.3, 0, 1, 0}, {6.3, 3.3, 6, 2.5, 0, 0, 1},   {5.8, 2.7, 5.1, 1.9, 0, 0, 1},
                      {7.1, 3, 5.9, 2.1, 0, 0, 1},   {6.3, 2.9, 5.6, 1.8, 0, 0, 1}, {6.5, 3, 5.8, 2.2, 0, 0, 1},
                      {7.6, 3, 6.6, 2.1, 0, 0, 1},   {4.9, 2.5, 4.5, 1.7, 0, 0, 1}, {7.3, 2.9, 6.3, 1.8, 0, 0, 1},
                      {6.7, 2.5, 5.8, 1.8, 0, 0, 1}, {7.2, 3.6, 6.1, 2.5, 0, 0, 1}, {6.5, 3.2, 5.1, 2, 0, 0, 1},
                      {6.4, 2.7, 5.3, 1.9, 0, 0, 1}, {6.8, 3, 5.5, 2.1, 0, 0, 1},   {5.7, 2.5, 5, 2, 0, 0, 1},
                      {5.8, 2.8, 5.1, 2.4, 0, 0, 1}, {6.4, 3.2, 5.3, 2.3, 0, 0, 1}, {6.5, 3, 5.5, 1.8, 0, 0, 1},
                      {7.7, 3.8, 6.7, 2.2, 0, 0, 1}, {7.7, 2.6, 6.9, 2.3, 0, 0, 1}, {6, 2.2, 5, 1.5, 0, 0, 1},
                      {6.9, 3.2, 5.7, 2.3, 0, 0, 1}, {5.6, 2.8, 4.9, 2, 0, 0, 1},   {7.7, 2.8, 6.7, 2, 0, 0, 1},
                      {6.3, 2.7, 4.9, 1.8, 0, 0, 1}, {6.7, 3.3, 5.7, 2.1, 0, 0, 1}, {7.2, 3.2, 6, 1.8, 0, 0, 1},
                      {6.2, 2.8, 4.8, 1.8, 0, 0, 1}, {6.1, 3, 4.9, 1.8, 0, 0, 1},   {6.4, 2.8, 5.6, 2.1, 0, 0, 1},
                      {7.2, 3, 5.8, 1.6, 0, 0, 1},   {7.4, 2.8, 6.1, 1.9, 0, 0, 1}, {7.9, 3.8, 6.4, 2, 0, 0, 1},
                      {6.4, 2.8, 5.6, 2.2, 0, 0, 1}, {6.3, 2.8, 5.1, 1.5, 0, 0, 1}, {6.1, 2.6, 5.6, 1.4, 0, 0, 1},
                      {7.7, 3, 6.1, 2.3, 0, 0, 1},   {6.3, 3.4, 5.6, 2.4, 0, 0, 1}, {6.4, 3.1, 5.5, 1.8, 0, 0, 1},
                      {6, 3, 4.8, 1.8, 0, 0, 1},     {6.9, 3.1, 5.4, 2.1, 0, 0, 1}, {6.7, 3.1, 5.6, 2.4, 0, 0, 1},
                      {6.9, 3.1, 5.1, 2.3, 0, 0, 1}, {5.8, 2.7, 5.1, 1.9, 0, 0, 1}, {6.8, 3.2, 5.9, 2.3, 0, 0, 1},
                      {6.7, 3.3, 5.7, 2.5, 0, 0, 1}, {6.7, 3, 5.2, 2.3, 0, 0, 1},   {6.3, 2.5, 5, 1.9, 0, 0, 1},
                      {6.5, 3, 5.2, 2, 0, 0, 1},     {6.2, 3.4, 5.4, 2.3, 0, 0, 1}, {5.9, 3, 5.1, 1.8, 0, 0, 1}});

  const int N_REGISTERS = raw_data.dimension(0);

  Eigen::Tensor<float, 2> iris_data(150, 7);

  std::vector<int> indexes(N_REGISTERS);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::shuffle(indexes.begin(), indexes.end(), rng);

  // Iris has a strong order bias. Shuffle it to avoid underfitting the last class

  Eigen::array<Eigen::Index, 2> row_extent = {1, iris_data.dimension(1)};

  for (int i = 0; i < N_REGISTERS; ++i) {
    int dest = indexes[i];
    Eigen::array<Eigen::Index, 2> A_offset = {i, 0};
    Eigen::array<Eigen::Index, 2> dest_offset = {dest, 0};
    const Eigen::Tensor<float, 2> row = raw_data.slice(A_offset, row_extent);
    iris_data.slice(dest_offset, row_extent) = row;
  }

  // 80% - training, 20% validation
  float split_percentage = .8;

  const int split_at = static_cast<int>(N_REGISTERS * split_percentage);

  Eigen::array<Eigen::Index, 2> training_x_offsets = {0, 0};
  Eigen::array<Eigen::Index, 2> training_x_extents = {split_at, 4};
  Eigen::Tensor<float, 2> training_X_ds = iris_data.slice(training_x_offsets, training_x_extents);

  Eigen::array<Eigen::Index, 2> training_y_offsets = {0, 4};
  Eigen::array<Eigen::Index, 2> training_y_extents = {split_at, 3};
  Eigen::Tensor<float, 2> training_Y_ds = iris_data.slice(training_y_offsets, training_y_extents);

  Eigen::array<Eigen::Index, 2> validation_x_offsets = {split_at, 0};
  Eigen::array<Eigen::Index, 2> validation_x_extents = {N_REGISTERS - split_at, 4};
  Eigen::Tensor<float, 2> validation_X_ds = iris_data.slice(validation_x_offsets, validation_x_extents);

  Eigen::array<Eigen::Index, 2> validation_y_offsets = {split_at, 4};
  Eigen::array<Eigen::Index, 2> validation_y_extents = {N_REGISTERS - split_at, 3};
  Eigen::Tensor<float, 2> validation_Y_ds = iris_data.slice(validation_y_offsets, validation_y_extents);

  std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
      result = std::make_tuple(training_X_ds, training_Y_ds, validation_X_ds, validation_Y_ds);

  return result;
}

/**
 * Batched version of Softmax
 */
template <typename T>
Eigen::Tensor<T, 2> softmax(const Eigen::Tensor<T, 2> &Z) {
  auto dimensions = Z.dimensions();

  const int batch_length = dimensions[0];
  const int instance_length = dimensions[1];

  const Eigen::array<int, 2> reshape_dim = {batch_length, 1};
  const Eigen::array<int, 2> bcast = {1, instance_length};

  Eigen::array<int, 1> depth_dim({1});
  auto z_max = Z.maximum(depth_dim);
  auto max_reshaped = z_max.reshape(reshape_dim);
  auto max_values = max_reshaped.broadcast(bcast);

  auto diff = Z - max_values;

  auto expo = diff.exp();
  auto expo_sums = expo.sum(depth_dim);
  auto sums_reshaped = expo_sums.reshape(reshape_dim);
  auto sums = sums_reshaped.broadcast(bcast);
  auto result = Eigen::Tensor<T, 2>(Z.dimension(0), Z.dimension(1));
  result = expo / sums;

  return result;
}

/**
 * Batched version of ReLU
 */
template <typename T>
Eigen::Tensor<T, 2> ReLU(const Eigen::Tensor<T, 2> &Z) {

  Eigen::Tensor<T, 2> result = Z.unaryExpr([](T v) {
    if (v > T(0.f)) return v;
    return T(0.f);
  });

  return result;
}

/**
 * Categorical Cross Entropy Cost function
 */
template <typename T>
T categorical_cross_entropy(const Eigen::Tensor<T, 2> &TRUE, const Eigen::Tensor<T, 2> &PRED) {

  auto loss = [](const T expected, const T actual) {
    T y_true = expected;
    T y_pred = actual;
    if (y_pred <= T(.0)) {
      y_pred = T(1e-7);
    } else if (y_pred >= T(1.)) {
      y_pred = T(1.);
    }
    T minus_true = -y_true;
    T result = minus_true * log(y_pred);
    return result;
  };

  Eigen::Tensor<T, 2> temp = TRUE.binaryExpr(PRED, loss);
  const Eigen::Tensor<T, 0> red = temp.sum();

  T result = red(0);

  return result;
}

/**
 * Convenience function to unpack the weight gradients from the LOSS autodiff
 */
auto unpack_gradients(const AutoDiff_T &LOSS, const Eigen::Tensor<AutoDiff_T, 2> &W0,
                      const Eigen::Tensor<AutoDiff_T, 2> &W1) {
  auto derivatives = LOSS.derivatives();

  int index = 0;
  Eigen::Tensor<float, 2> grad0(W0.dimension(0), W0.dimension(1));
  for (int i = 0; i < W0.dimension(0); ++i) {
    for (int j = 0; j < W0.dimension(1); ++j) {
      float val = derivatives[index];
      grad0(i, j) = val;
      index++;
    }
  }

  Eigen::Tensor<float, 2> grad1(W1.dimension(0), W1.dimension(1));
  for (int i = 0; i < W1.dimension(0); ++i) {
    for (int j = 0; j < W1.dimension(1); ++j) {
      float val = derivatives[index];
      grad1(i, j) = val;
      index++;
    }
  }

  return std::make_pair(grad0, grad1);
}

/**
 * Utility to convert raw-scalar tensors in AutoDiff-tensors
 */
Eigen::Tensor<AutoDiff_T, 2> convert(const Eigen::Tensor<float, 2> &T, int size = 0, int offset = 0) {
  Eigen::Tensor<AutoDiff_T, 2> result = T.cast<AutoDiff_T>();
  if (size) {
    int pos = offset;
    const size_t m = T.dimension(0);
    const size_t n = T.dimension(1);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result(i, j).value() = T(i, j);
        result(i, j).derivatives() = Eigen::VectorXf::Unit(size, pos++);
      }
    }
  }
  return result;
}

/**
 * Accuracy metric
 */
template <typename T, int RANK_>
float accuracy(Eigen::Tensor<T, RANK_> &REAL, Eigen::Tensor<T, RANK_> &PRED) {

  auto compare = [](int a, int b) { return static_cast<float>(a == b); };

  Eigen::Tensor<Eigen::DenseIndex, RANK_ - 1> REAL_MAX = REAL.argmax(RANK_ - 1);
  Eigen::Tensor<Eigen::DenseIndex, RANK_ - 1> PRED_MAX = PRED.argmax(RANK_ - 1);
  auto diff = REAL_MAX.binaryExpr(PRED_MAX, compare);

  Eigen::Tensor<float, 0> mean = diff.mean();

  float result = mean(0) * 100.f;

  return result;
}

/**
 * Training loop
 */
template <typename T, typename Device>
T loop(const Device &device, const Eigen::Tensor<T, 2> &_TRUE, const Eigen::Tensor<T, 2> &_X, Eigen::Tensor<T, 2> &_W0,
       Eigen::Tensor<T, 2> &_W1, T learning_rate) {

  // converting tensors to autodiff
  Eigen::Tensor<AutoDiff_T, 2> TRUE = convert(_TRUE);
  Eigen::Tensor<AutoDiff_T, 2> X = convert(_X);
  Eigen::Tensor<AutoDiff_T, 2> W0 = convert(_W0, _W0.size() + _W1.size());
  Eigen::Tensor<AutoDiff_T, 2> W1 = convert(_W1, _W0.size() + _W1.size(), W0.size());

  // forward pass
  const Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

  // Hidden layer
  Eigen::Tensor<AutoDiff_T, 2> Z0(X.dimension(0), W0.dimension(1));
  Z0.device(device) = X.contract(W0, contract_dims);
  Eigen::Tensor<AutoDiff_T, 2> Y0 = ReLU(Z0);

  // Output Layer
  Eigen::Tensor<AutoDiff_T, 2> Z1(Y0.dimension(0), W1.dimension(1));
  Z1 = Y0.contract(W1, contract_dims);
  auto Y1 = softmax(Z1);

  AutoDiff_T LOSS = categorical_cross_entropy(TRUE, Y1);

  // backward pass
  auto gradients = unpack_gradients(LOSS, W0, W1);

  auto grad0 = std::get<0>(gradients);
  auto grad1 = std::get<1>(gradients);

  // update pass
  _W0 = _W0 - grad0 * grad0.constant(learning_rate);
  _W1 = _W1 - grad1 * grad1.constant(learning_rate);

  T result = LOSS.value();

  return result;
}

/**
 * Function to solve the predictions given an batch of inputs
 */
template <typename T>
Eigen::Tensor<T, 2> predict(const Eigen::Tensor<T, 2> &X, Eigen::Tensor<T, 2> &W0, Eigen::Tensor<T, 2> &W1) {
  const Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

  Eigen::Tensor<T, 2> Z0 = X.contract(W0, contract_dims);
  Eigen::Tensor<T, 2> Y0 = ReLU(Z0);

  Eigen::Tensor<T, 2> Z1 = Y0.contract(W1, contract_dims);
  auto Y1 = softmax(Z1);

  return Y1;
}

/**
 * Uniform Glorot weight initialization
 */
Eigen::Tensor<float, 2> parameter_initializer(int rows, int cols) {
  float range = sqrt(6.f / (rows + cols));
  std::uniform_real_distribution<float> uniform_distro(-range, range);
  Eigen::Tensor<float, 2> result(rows, cols);
  result = result.unaryExpr([&uniform_distro](float) { return uniform_distro(rng); });
  return result;
}

int main(int, char **) {
  auto data = load_iris_dataset();

  auto training_X_ds = std::get<0>(data);
  auto training_Y_ds = std::get<1>(data);
  auto validation_X_ds = std::get<2>(data);
  auto validation_Y_ds = std::get<3>(data);

  std::cout << "Data loaded!\n";

  std::cout << "training input dim: " << training_X_ds.dimensions() << "\n\n";
  std::cout << "training labels dim: " << training_Y_ds.dimensions() << "\n\n";

  std::cout << "validation input dim: " << validation_X_ds.dimensions() << "\n\n";
  std::cout << "validation labels dim: " << validation_Y_ds.dimensions() << "\n\n";

  const int training_size = training_X_ds.dimension(0);
  const int input_size = training_X_ds.dimension(1);
  const int output_size = training_Y_ds.dimension(1);

  const int hidden_units = 32;
  const int batch_size = 20;

  Eigen::Tensor<float, 2> W0 = parameter_initializer(input_size, hidden_units);
  Eigen::Tensor<float, 2> W1 = parameter_initializer(hidden_units, output_size);

  const int MAX_EPOCHS = 1000;
  const int VERBOSE_EACH = MAX_EPOCHS / 20;
  const float learning_rate = 0.0001f;

  int epoch = 0;
  const int steps = training_size / batch_size;

  Eigen::DefaultDevice device;

  std::vector<std::pair<float, float>> training_losses;
  std::vector<std::pair<float, float>> validation_losses;
  std::vector<std::pair<float, float>> training_accs;
  std::vector<std::pair<float, float>> validation_accs;
  while (epoch++ < MAX_EPOCHS) {
    float training_loss = 0.f;

    for (int step = 0; step < steps; ++step) {
      int index = step * batch_size;
      int batch = std::min(training_size - index, batch_size);

      Eigen::array<Eigen::Index, 2> offset = {index, 0};
      const Eigen::array<Eigen::Index, 2> input_extent = {batch, input_size};
      const Eigen::array<Eigen::Index, 2> output_extent = {batch, output_size};

      Eigen::Tensor<float, 2> X_batch = training_X_ds.slice(offset, input_extent);
      Eigen::Tensor<float, 2> Y_batch = training_Y_ds.slice(offset, output_extent);

      float loss = loop(device, Y_batch, X_batch, W0, W1, learning_rate);
      training_loss += loss;
    }

    {
      auto validation_pred = predict(validation_X_ds, W0, W1);
      float validation_loss = categorical_cross_entropy(validation_Y_ds, validation_pred);
      float validation_acc = accuracy(validation_Y_ds, validation_pred);

      training_loss /= training_size;
      validation_loss /= validation_Y_ds.dimension(0);

      auto training_pred = predict(training_X_ds, W0, W1);
      float training_acc = accuracy(training_Y_ds, training_pred);

      if (epoch % VERBOSE_EACH == 0) {
        std::cout << "epoch:\t" << epoch << "\ttraining_loss:\t" << training_loss << "\tvalidation_loss:\t"
                  << validation_loss << "\ttraining_acc:\t" << training_acc << "\tvalidation_acc:\t" << validation_acc
                  << "\n";
      }

      training_losses.emplace_back(std::make_pair(epoch, training_loss));
      validation_losses.emplace_back(std::make_pair(epoch, validation_loss));
    }
  }

  return 0;
}