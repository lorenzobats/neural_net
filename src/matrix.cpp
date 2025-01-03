#include "matrix.h"
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

Matrix::Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
  values.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(const std::vector<std::vector<double>> &values)
    : rows(values.size()), cols(values[0].size()), values(values) {}

void Matrix::debug() {
  for (int i = 0; i < this->getRows(); ++i) {
    for (int j = 0; j < this->getCols(); ++j) {
      std::cout << this->getValues().at(i).at(j);
      if (j < this->getCols() - 1) {
        std::cout << ",";
      }
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::transpose() const {
  Matrix transposed(this->cols, this->rows);
  for (int i = 0; i < this->cols; ++i) {
    for (int j = 0; j < this->rows; ++j) {
      transposed(i, j) = (*this)(j, i);
    }
  }
  return transposed;
}

Matrix Matrix::sumRows() const {
  Matrix result(this->rows, 1);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      result(i, 0) += (*this)(i, j);
    }
  }
  return result;
}

Matrix Matrix::operator+(const double n) const {
  Matrix result(rows, cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      result(i, j) = (*this)(i, j) + n;
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  if (cols != other.cols || rows != other.rows) {
    throw std::invalid_argument(
        "Error while subtracting Matrices. Incompatible dimensions A (" +
        std::to_string(getRows()) + ", " + std::to_string(getCols()) +
        ") and B (" + std::to_string(other.getRows()) + +", " +
        std::to_string(other.getCols()) + ")");
  }

  Matrix matrix(this->rows, other.cols);

  for (int i = 0; i < this->rows; ++i) {
    for (int k = 0; k < other.cols; ++k) {
      for (int j = 0; j < this->cols; ++j) {
        matrix(i, k) += (*this)(i, j) - other(j, k);
      }
    }
  }

  return matrix;
}

Matrix Matrix::operator-(const double n) const {
  Matrix result(rows, cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      result(i, j) = (*this)(i, j) - n;
    }
  }
  return result;
}

Matrix Matrix::operator*(const double factor) const {
  Matrix result(rows, cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      result(i, j) = (*this)(i, j) * factor;
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  if (this->cols != other.rows) {
    throw std::invalid_argument(
        "Matrix multiplication error: A(" + std::to_string(this->rows) + "x" +
        std::to_string(this->cols) + ") * B(" + std::to_string(other.rows) +
        "x" + std::to_string(other.cols) +
        ") is invalid. Columns of A must equal rows of B.");
  }

  Matrix matrix(this->rows, other.cols);

  for (int i = 0; i < this->rows; ++i) {
    for (int k = 0; k < other.cols; ++k) {
      for (int j = 0; j < this->cols; ++j) {
        matrix(i, k) += (*this)(i, j) * other(j, k);
      }
    }
  }

  return matrix;
}

double &Matrix::operator()(int row, int col) { return values[row][col]; }

const double &Matrix::operator()(int row, int col) const {
  return values[row][col];
}

const std::vector<std::vector<double>> Matrix::getValues() const {
  return this->values;
}

const int Matrix::getRows() const { return this->rows; }

const int Matrix::getCols() const { return this->cols; }
