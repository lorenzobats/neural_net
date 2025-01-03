#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include "layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizese);

    void train(const Matrix& input, const Matrix& target, const double learning_rate, const int epochs);

	void debug();

private:
    std::vector<Layer> layers;
    Matrix calculateLoss();
    void backpropagate();
};

#endif
