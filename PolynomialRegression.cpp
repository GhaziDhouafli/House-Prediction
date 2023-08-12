#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

// Function to perform linear regression using gradient descent
void linearRegressionGradientDescent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>& coefficients, double learningRate, int numIterations) {
    int m = X.size(); // Number of training examples
    int n = X[0].size(); // Number of features
    std::vector<double> gradients(n, 0.0);

    for (int i = 0; i < numIterations; ++i) {
        double cost = 0.0;

        for (int j = 0; j < m; ++j) {
            double yPredicted = 0.0;
            for (int k = 0; k < n; ++k) {
                yPredicted += coefficients[k] * X[j][k];
            }
            cost += (yPredicted - y[j]) * (yPredicted - y[j]);

            for (int k = 0; k < n; ++k) {
                gradients[k] += 2 * (yPredicted - y[j]) * X[j][k];
            }
        }

        cost /= m;

        for (int k = 0; k < n; ++k) {
            gradients[k] /= m;
            coefficients[k] -= learningRate * gradients[k];
        }

        std::cout << "Iteration " << i + 1 << ", Cost: " << cost << std::endl;
    }
}

int main() {
    std::ifstream file("house_price.csv");

    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    std::vector<double> targetVector;
    std::vector<std::vector<double>> inputMatrix;
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string col;
        std::vector<double> row;

        for (int i = 0; std::getline(iss, col, ','); ++i) {
            double value = std::stod(col);

            if (i == 2) {
                // Store the target value in the targetVector
                targetVector.push_back(value);
            } else {
                // Store other values in the inputMatrix
                row.push_back(value);
            }
        }

        inputMatrix.push_back(row);
    }

    // Create polynomial features
    int degree = 2;  // Set the degree of the polynomial
    std::vector<std::vector<double>> polyInputMatrix;
    for (const std::vector<double>& row : inputMatrix) {
        std::vector<double> polyRow;
        for (int d = 0; d <= degree; ++d) {
            for (double value : row) {
                polyRow.push_back(std::pow(value, d));
            }
        }
        polyInputMatrix.push_back(polyRow);
    }

    // Initial coefficients
    std::vector<double> coefficients(polyInputMatrix[0].size(), 0.0);

    // Hyperparameters
    double learningRate = 0.01;
    int numIterations = 100;

    linearRegressionGradientDescent(polyInputMatrix, targetVector, coefficients, learningRate, numIterations);

    return 0;
}
