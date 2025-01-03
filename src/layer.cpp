#include "layer.h"
#include "matrix.h"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>

Layer::Layer(const int input_size, const int output_size)
    : input_size(input_size), output_size(output_size),
      weights(input_size, output_size), biases(output_size, 1),
      last_input(1, input_size) {

  std::srand(std::time(nullptr));

  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < output_size; ++j) {
      weights(i, j) = (std::rand() % 2000 - 1000) / 1000.0;
    }
  }

  for (int i = 0; i < output_size; ++i) {
    biases(i, 0) = (std::rand() % 2000 - 1000) / 1000.0;
  }
}

double Layer::sigmoid(const double input) { return 1 / (1 + exp(-input)); }

Matrix Layer::forward(const Matrix &input) {
  Matrix z = input * weights;

  // add biases
  for (int i = 0; i < z.getRows(); ++i) {
    for (int j = 0; j < z.getCols(); ++j) {
      z(i, j) += biases(j, 0);
    }
  }

  for (int i = 0; i < z.getRows(); ++i) {
    for (int j = 0; j < z.getCols(); ++j) {
      z(i, j) = sigmoid(z(i, j));
    }
  }

  z.debug();

  return z;
}

Matrix Layer::backward(const Matrix &grad_output, const double learning_rate) {
  std::cout << "Calculating Backward" << std::endl;

  // Calculate gradients
  Matrix grad_weights = grad_output.transpose() * this->last_input;
  Matrix grad_biases = grad_output.sumRows();
  Matrix grad_input = grad_output * weights.transpose();

  // Update weights and biases
  updateWeights(grad_weights, learning_rate);
  updateBiases(grad_biases, learning_rate);

  return grad_input;
}

void Layer::updateWeights(const Matrix &grad_weights,
                          const double learning_rate) {
  this->weights = this->weights - grad_weights * learning_rate;
}

void Layer::updateBiases(const Matrix &grad_biases,
                         const double learning_rate) {
  this->biases = this->biases - grad_biases * learning_rate;
}

void Layer::debug() {
  std::cout << "Input size: " << input_size << std::endl;
  std::cout << "Output size: " << output_size << std::endl;
}
