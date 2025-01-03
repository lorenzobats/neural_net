#ifndef MATRIX_H
#define MATRIX_H 

#include <vector>

class Matrix {
public:
	Matrix(const int rows, const int cols);
	Matrix(const std::vector<std::vector<double>>& values);

	Matrix transpose() const;
	Matrix sumRows() const;

	Matrix operator+(const Matrix& other) const;
	Matrix operator+(const double n) const;

	Matrix operator-(const Matrix& other) const;
	Matrix operator-(const double n) const;

	Matrix operator*(const Matrix& other) const;
	Matrix operator*(const double n) const;

	double& operator()(int row, int col); // element access
	const double& operator()(int row, int col) const; // read access

	const std::vector<std::vector<double>> getValues() const;
	const int getRows() const;
	const int getCols() const;
	void debug();

private:
	int rows;
	int cols;
	std::vector<std::vector<double>> values;
};

#endif
