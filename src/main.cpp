#include <iostream>
#include "../include/Spiral.h"
#include "../include/DenseLayer.h"
#include "../include/ActivationReLu.h"
#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// using namespace std::chrono;
// auto start = high_resolution_clock::now();
// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << duration.count() << std::endl;

int main() {
    Spiral dataset(100, 3);
    DenseLayer layer1(2, 3);
    ActivationReLu activation1 = ActivationReLu();
    DenseLayer layer2(3, 3);
    ActivationSoftmax activation2 = ActivationSoftmax();
    LossCategoricalCrossEntropy loss = LossCategoricalCrossEntropy();

    layer1.forward(dataset.getX());
    activation1.forward(layer1.getOutput());
    layer2.forward(activation1.getOutput());
    activation2.forward(layer2.getOutput());

    // cout << activation2.getOutput() << endl;

    // cout << loss.forward(activation2.getOutput(), dataset.getY()) << endl;
    // cout << loss.calculate(activation2.getOutput(), dataset.getY()) << endl;

    // Eigen::MatrixXd softmaxOutputs = activation2.getOutput();
    // Eigen::VectorXd predictions = Eigen::VectorXd::Zero(softmaxOutputs.rows());
    // Eigen::Index maxRow;
    // for(int rowIndex = 0; rowIndex < predictions.rows(); rowIndex++) {
    //     softmaxOutputs.row(rowIndex).maxCoeff(&maxRow);
    //     predictions(rowIndex) = maxRow;
    // }

    // int matchCount = 0;
    // for(int rowIndex = 0; rowIndex < predictions.rows(); rowIndex++) {
    //     if(predictions(rowIndex) == dataset.getY()(rowIndex)) {
    //         matchCount++;
    //     }
    // }

    // cout << (double) matchCount / dataset.getY().rows() << endl;

    return 0;
}