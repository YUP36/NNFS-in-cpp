#include <iostream>
#include "../include/Spiral.h"
#include "../include/DenseLayer.h"
#include "../include/ActivationReLu.h"
#include "../include/ActivationSoftmax.h"
#include "../include/LossCategoricalCrossEntropy.h"
#include "../include/ActivationSoftmaxLossCategoricalCrossEntropy.h"
#include "../include/OptimizerSGD.h"

// #include <omp.h>
//  /opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
#include "/opt/homebrew/opt/libomp/include/omp.h"

//   export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
//   export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using namespace std::chrono;


// auto start = high_resolution_clock::now();

// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << duration.count() << std::endl;

int main() {
    omp_set_num_threads(4);
    // Eigen::setNbThreads(2);
    cout << Eigen::nbThreads() << endl;

    Spiral dataset(100, 3);
    MatrixXd X = dataset.getX();
    VectorXi Y = dataset.getY();

    DenseLayer layer1(2, 64);
    ActivationReLu activation1 = ActivationReLu();
    DenseLayer layer2(64, 3);
    // ActivationSoftmax activation2 = ActivationSoftmax();
    // LossCategoricalCrossEntropy CCE = LossCategoricalCrossEntropy();
    ActivationSoftmaxLossCategoricalCrossEntropy activationLoss = ActivationSoftmaxLossCategoricalCrossEntropy();
    OptimizerSGD optimizer = OptimizerSGD();

    double loss;
    MatrixXd softmaxOutputs;
    VectorXd predictions;
    int matchCount;
    Eigen::Index maxRow;
    for(int epoch = 0; epoch < 10001; epoch++){

        auto start = high_resolution_clock::now();
        // forward pass: 1792
        layer1.forward(&X);
        activation1.forward(layer1.getOutput());
        layer2.forward(activation1.getOutput());
        // activation2.forward(layer2.getOutput());
        // loss = CCE.calculate(layer2.getOutput(), &Y);
        loss = activationLoss.forwardAndCalculate(layer2.getOutput(), &Y);

        // accuracy calulation: 72
        softmaxOutputs = *(activationLoss.getOutput());
        predictions = Eigen::VectorXd::Zero(softmaxOutputs.rows());
        for(int rowIndex = 0; rowIndex < predictions.rows(); rowIndex++) {
            softmaxOutputs.row(rowIndex).maxCoeff(&maxRow);
            predictions(rowIndex) = maxRow;
        }

        matchCount = 0;
        for(int rowIndex = 0; rowIndex < predictions.rows(); rowIndex++) {
            if(predictions(rowIndex) == Y(rowIndex)) {
                matchCount++;
            }
        }

        // // backward pass: ~2700
        // CCE.backward(activation2.getOutput(), &Y);
        // activation2.backward(CCE.getDinputs());
        activationLoss.backward(activationLoss.getOutput(), &Y);
        layer2.backward(activationLoss.getDinputs());
        activation1.backward(layer2.getDinputs());
        layer1.backward(activation1.getDinputs());

        // parameter update: 14
        optimizer.updateParameters(&layer1);
        optimizer.updateParameters(&layer2);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        if((epoch % 100) == 0) {
            cout << "Epoch: " << epoch << "\t";
            cout << "Loss: " << loss << "\t";
            cout << "Accuracy: " << (double) matchCount / Y.rows() << endl;
            std::cout << duration.count() << std::endl;
        }
    }

    return 0;
}