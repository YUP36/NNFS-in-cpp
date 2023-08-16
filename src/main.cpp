#include <iostream>
#include <vector>

#include "../include/DataGeneration/lodepng.h"
#include "../include/DataGeneration/ImageGenerator.h"
#include "../include/DataGeneration/Spiral.h"
#include "../include/DataGeneration/Sine.h"

#include "../include/Layers/Dense.h"
#include "../include/Layers/Dropout.h"

#include "../include/ActivationFunctions/ReLu.h"
#include "../include/ActivationFunctions/Softmax.h"
#include "../include/ActivationFunctions/Sigmoid.h"
#include "../include/ActivationFunctions/Linear.h"

#include "../include/LossFunctions/CategoricalCrossEntropy.h"
#include "../include/LossFunctions/SoftmaxCategoricalCrossEntropy.h"
#include "../include/LossFunctions/BinaryCrossEntropy.h"
#include "../include/LossFunctions/MeanAbsoluteError.h"
#include "../include/LossFunctions/MeanSquaredError.h"

#include "../include/Optimizers/SGD.h"
#include "../include/Optimizers/Adagrad.h"
#include "../include/Optimizers/RMSProp.h"
#include "../include/Optimizers/Adam.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::RowVectorXd;
using namespace std::chrono;

// auto start = high_resolution_clock::now();

// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << duration.count() << std::endl;

int main() {
    srand(1);

    Sine dataset = Sine();
    MatrixXd X = dataset.getX();
    MatrixXd Y = dataset.getY();

    Dense dense1 = Dense(1, 64);
    ReLu activation1 = ReLu();
    Dense dense2 = Dense(64, 64);
    ReLu activation2 = ReLu();
    Dense dense3 = Dense(64, 1);
    Linear activation3 = Linear();
    MeanSquaredError lossFunction = MeanSquaredError();

    Adam optimizer = Adam(0.003, 1e-3);
    
    double standardDeviation = (Y.array() - Y.mean()).square().sum() / Y.rows();
    double precision = standardDeviation / 250.0;
    cout << precision << endl;
    double loss, dataLoss, regularizationLoss;
    MatrixXd outputs;
    // VectorXd predictions = VectorXd::Zero(200);
    int matchCount;
    // Eigen::Index maxRow;
    for(int epoch = 0; epoch < 10001; epoch++){

        // auto start = high_resolution_clock::now();
        // FORWARD PASS
        dense1.forward(&X);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());
        dense3.forward(activation2.getOutput());
        activation3.forward(dense3.getOutput());
        lossFunction.forward(activation3.getOutput(), &Y);

        dataLoss = lossFunction.calculate(activation3.getOutput(), &Y);
        regularizationLoss = lossFunction.calculateRegularizationLoss(&dense1)
                            + lossFunction.calculateRegularizationLoss(&dense2)
                            + lossFunction.calculateRegularizationLoss(&dense3);
        loss = dataLoss + regularizationLoss;

        // ACCURACY CALCULATION
        outputs = *(activation3.getOutput());
        MatrixXd difference = (outputs - Y).array().abs();
        matchCount = difference.unaryExpr([precision](double x){return (x < precision) ? 1.0 : 0.0;}).sum();

        // BACKPROPAGATION
        lossFunction.backward(activation3.getOutput(), &Y);
        activation3.backward(lossFunction.getDinputs());
        dense3.backward(activation3.getDinputs());
        activation2.backward(dense3.getDinputs());
        dense2.backward(activation2.getDinputs());
        activation1.backward(dense2.getDinputs());
        dense1.backward(activation1.getDinputs());

        // PARAMETER UPDATE
        optimizer.decay();
        optimizer.updateParameters(&dense1);
        optimizer.updateParameters(&dense2);
        optimizer.updateParameters(&dense3);
        optimizer.incrementIteration();

        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);

        if((epoch % 100) == 0) {
            cout << "Epoch: " << epoch << "\t";
            cout << "Loss: " << loss << "\t";
            cout << "Data loss: " << dataLoss << "\t";
            cout << "Regularization loss: " << regularizationLoss << "\t";
            cout << "Accuracy: " << (double) matchCount / Y.rows() << endl;
            // cout << optimizer.getLearningRate() << endl;
            // std::cout << duration.count() << std::endl;
        }
    }

    //////////////////////////////////////////////////////////////////////
    ///////////////////////////// TEST DATA //////////////////////////////

    Spiral test(100, 2);
    MatrixXd XTest = test.getX();
    MatrixXd YTest = test.getY();

    dense1.forward(&X);
    activation1.forward(dense1.getOutput());
    dense2.forward(activation1.getOutput());
    activation2.forward(dense2.getOutput());
    dense3.forward(activation2.getOutput());
    activation3.forward(dense3.getOutput());
    lossFunction.forward(activation3.getOutput(), &Y);
    
    dataLoss = lossFunction.calculate(activation3.getOutput(), &Y);
    regularizationLoss = lossFunction.calculateRegularizationLoss(&dense1)
                        + lossFunction.calculateRegularizationLoss(&dense2)
                        + lossFunction.calculateRegularizationLoss(&dense3);
    loss = dataLoss + regularizationLoss;

    outputs = *(activation3.getOutput());
    MatrixXd difference = (outputs - Y).array().abs();
    matchCount = difference.unaryExpr([precision](double x){return (x < precision) ? 1.0 : 0.0;}).sum();
    
    cout << "Test Data: \t Loss: " << loss << "\t";
    cout << "Accuracy: " << (double) matchCount / Y.rows() << endl;
    // cout << optimizer.getLearningRate() << endl;
    
    //////////////////////////////////////////////////////////////////////////
    ///////////////////////////// VISUALIZATION //////////////////////////////
    // const int WIDTH = 1000;
    // const int HEIGHT = 1000;
    // MatrixXd inputGrid = MatrixXd(WIDTH * HEIGHT, 2);
    // for(int y = 0; y < HEIGHT; y++) {
    //     for(int x = 0; x < WIDTH; x++) {
    //         inputGrid((y * HEIGHT) + x, 0) = (x * 2.0 / WIDTH) - 1;
    //         inputGrid((y * HEIGHT) + x, 1) = (y * 2.0 / HEIGHT) - 1;
    //     }
    // }

    // dense1.forward(&inputGrid);
    // activation1.forward(dense1.getOutput());
    // dense2.forward(activation1.getOutput());
    // activation2.forward(dense2.getOutput());

    // std::vector<unsigned char> pixels(WIDTH * HEIGHT * 4); // RGBA format
    // double pix;
    // for (int y = 0; y < HEIGHT; ++y) {
    //     for (int x = 0; x < WIDTH; ++x) {
    //         int index = 4 * (y * WIDTH + x);
    //         pix = (*(activation2.getOutput()))(y * HEIGHT + x, 0);
    //         // green: 109, 209, 129
    //         // red: 255, 130, 130
    //         pixels[index + 0] = sqrt(109 * 109 * pix + 255 * 255 * (1 - pix));
    //         pixels[index + 1] = sqrt(209 * 209 * pix + 130 * 130 * (1 - pix));
    //         pixels[index + 2] = sqrt(129 * 129 * pix + 130 * 130 * (1 - pix));
    //         pixels[index + 3] = 255; // Alpha channel (opacity: 255 = fully opaque)
    //     }
    // }
    // ImageGenerator gen = ImageGenerator();
    // gen.createImage(pixels, "visualizations/adam/binaryCrossEntropylr0.01dr5e-7.png", WIDTH, HEIGHT);

    return 0;
}