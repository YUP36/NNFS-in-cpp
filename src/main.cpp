#include <iostream>
#include "../include/Spiral.h"
#include "../include/DenseLayer.h"
#include "../include/ActivationReLu.h"
#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"
#include "../include/ActivationSoftmaxLossCategoricalCrossEntropy.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std::chrono;
// auto start = high_resolution_clock::now();
// loss.backward(softmax_outputs, class_targets);
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
    ActivationSoftmaxLossCategoricalCrossEntropy softmaxLoss = ActivationSoftmaxLossCategoricalCrossEntropy();

    // MatrixXd softmax_outputs(3,3);
    // softmax_outputs << 0.7, 0.1, 0.2,
    //                 0.1, 0.5, 0.4,
    //                 0.02, 0.9, 0.08;
    // VectorXi class_targets(3);
    // class_targets << 0, 1, 1;

    // loss.backward(softmax_outputs, class_targets);
    // activation2.output = softmax_outputs;
    // activation2.backward(loss.getDinputs());
    // cout << activation2.getDinputs() << endl;

    // softmaxLoss.backward(softmax_outputs, class_targets);
    // cout << softmaxLoss.getDinputs() << endl;


    layer1.forward(dataset.getX());
    activation1.forward(layer1.getOutput());
    layer2.forward(activation1.getOutput());
    activation2.forward(layer2.getOutput());

    loss.backward(activation2.getOutput(), dataset.getY());
    activation2.backward(loss.getDinputs());

    softmaxLoss.backward(activation2.getOutput(), dataset.getY());

    // cout << loss.forward(activation2.getOutput(), dataset.getY()) << endl;
    // cout << endl << loss.calculate(activation2.getOutput(), dataset.getY()) << endl;

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