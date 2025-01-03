#include <iostream>
#include <fstream>
#include <cstdint>
#include<stdexcept>
#include <vector>

#include "neural_network.h"
#include "matrix.h"

std::vector<std::vector<uint8_t>> load_images(const std::string& file_path) {
	std::ifstream file(file_path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open file " + file_path);
	}

	int32_t magic_number = 0;
	int32_t num_images = 0;
	int32_t num_rows = 0;
	int32_t num_cols = 0;

	file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);

	file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = __builtin_bswap32(num_images);

	file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = __builtin_bswap32(num_rows);

	file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = __builtin_bswap32(num_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid magic number for image file.");
    }

	std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_cols * num_rows));
	for (int i = 0; i < num_images; ++i) {
		file.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
	}

	return images;
}

std::vector<uint8_t> load_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    int32_t magic_number = 0;
    int32_t num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);

    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid magic number for label file.");
    }

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

int main() {
	std::vector<int> layer_sizes = {2, 10, 10, 2};
	NeuralNetwork network(layer_sizes);

	Matrix input({{0.9, 0.3}});
	Matrix target({{0., 0.8}});
	network.debug();
	network.train(input, target, 0.0001, 10);

    try {
        auto train_images = load_images("dataset/train-images.idx3-ubyte");
        auto train_labels = load_labels("dataset/train-labels.idx1-ubyte");

        std::cout << "Loaded " << train_images.size() << " training images and "
                  << train_labels.size() << " labels." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
	
	std::vector<std::vector<double>> matA = {{1, 2, 3}, {4, 5, 6}};
	std::vector<std::vector<double>> matB = {{9, 8, 7, 6}, {5, 4, 3, 2}, {1, 0, 0, 0}};

	Matrix A(matA);
	Matrix B(matB);
	Matrix result = A * B;
	Matrix sumRows = result.sumRows();
	sumRows.debug();
	return 0;
}

