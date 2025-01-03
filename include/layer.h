#ifndef LAYER_H
#define LAYER_H

#include "matrix.h" 

class Layer {
public:
	Layer(const int input_size, const int output_size);
	
	Matrix forward(const Matrix& input);
	Matrix backward(const Matrix& input, const double learning_rate);

	void updateWeights(const Matrix& weights, const double learning_rate); 
	void updateBiases(const Matrix& biases, const double learning_rate); 

	const Matrix& getBiases() const;
	const Matrix& getWeights() const;

	double sigmoid(const double input);
	void debug();
private:
	int input_size;
	int output_size;

	Matrix weights;
	Matrix biases;
	Matrix last_input;
};

#endif
