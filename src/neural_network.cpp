#include "neural_network.h"
#include <iostream>
#include <math.h>
#include <vector>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layerSizes) {
  if (layerSizes.size() < 2) {
    throw std::invalid_argument(
        "Neural network must have at least two layers (input and output).");
  }

  for (size_t i = 1; i < layerSizes.size(); ++i) {
    layers.emplace_back(Layer(layerSizes[i - 1], layerSizes[i]));
  }
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target,
                          const double learning_rate, const int epochs) {
  Matrix next = input;
  for (int n = 0; n < epochs; ++n) {
    std::cout << "===== Epoch: " << n + 1 << " =====" << std::endl;
    for (Layer &layer : layers) {
      next = layer.forward(next);
    }

    // calculate loss
    double loss = 0.0;
    Matrix grad_loss(next.getRows(), next.getCols());
    for (int i = 0; i < next.getRows(); ++i) {
      for (int j = 0; j < next.getCols(); ++j) {
        double error = target(i, j) - next(i, j);
        loss += 0.5 * pow(error, 2);
        grad_loss(i, j) = -error;
      }
    }

    std::cout << "Loss: " << loss << std::endl;

    // backpropagation
    Matrix gradient = grad_loss;
    for (Layer &layer : layers) {
      gradient = layer.backward(gradient, learning_rate);
    }
  }
}

void NeuralNetwork::debug() {
  for (size_t i = 0; i < layers.size(); ++i) {
    std::cout << "====== Layer: " << i << " ======" << std::endl;
    layers.at(i).debug();
    std::cout << "======================" << std::endl;
  }
}
