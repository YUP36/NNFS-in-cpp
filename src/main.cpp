#include <iostream>
#include <vector>

#include "../include/lodepng.h"
#include "../include/ImageGenerator.h"
#include "../include/Spiral.h"

#include "../include/DenseLayer.h"

#include "../include/ActivationReLu.h"
#include "../include/ActivationSoftmax.h"

#include "../include/LossCategoricalCrossEntropy.h"
#include "../include/ActivationSoftmaxLossCategoricalCrossEntropy.h"

#include "../include/OptimizerSGD.h"

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
    VectorXd predictions = VectorXd::Zero(300);
    int matchCount;
    Eigen::Index maxRow;
    for(int epoch = 0; epoch < 10001; epoch++){

        // auto start = high_resolution_clock::now();

        // FORWARD PASS
        layer1.forward(&X);
        activation1.forward(layer1.getOutput());
        layer2.forward(activation1.getOutput());
        // activation2.forward(layer2.getOutput());
        // loss = CCE.calculate(layer2.getOutput(), &Y);
        activationLoss.forward(layer2.getOutput());
        loss = activationLoss.calculate(&Y);

        // ACCURACY CALCULATION
        softmaxOutputs = *(activationLoss.getOutput());
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

        // BACKPROPAGATION
        // CCE.backward(activation2.getOutput(), &Y);
        // activation2.backward(CCE.getDinputs());
        activationLoss.backward(activationLoss.getOutput(), &Y);
        layer2.backward(activationLoss.getDinputs());
        activation1.backward(layer2.getDinputs());
        layer1.backward(activation1.getDinputs());

        // PARAMETER UPDATE
        optimizer.decay();
        optimizer.updateParameters(&layer1);
        optimizer.updateParameters(&layer2);
        optimizer.incrementIteration();

        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);

        if((epoch % 100) == 0) {
            cout << "Epoch: " << epoch << "\t";
            cout << "Loss: " << loss << "\t";
            cout << "Accuracy: " << (double) matchCount / Y.rows() << endl;
            cout << optimizer.getLearningRate() << endl;
            // std::cout << duration.count() << std::endl;
        }
    }


    // VISUALIZATION
    const int WIDTH = 1000;
    const int HEIGHT = 1000;
    MatrixXd inputGrid = MatrixXd(WIDTH * HEIGHT, 2);
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            inputGrid((y * HEIGHT) + x, 0) = (x * 2.0 / WIDTH) - 1;
            inputGrid((y * HEIGHT) + x, 1) = (y * 2.0 / HEIGHT) - 1;
        }
    }

    layer1.forward(&inputGrid);
    activation1.forward(layer1.getOutput());
    layer2.forward(activation1.getOutput());
    activationLoss.forward(layer2.getOutput());

    std::vector<unsigned char> pixels(WIDTH * HEIGHT * 4); // RGBA format
    RowVectorXd pix = RowVectorXd(1, 3);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int index = 4 * (y * WIDTH + x);
            pix = activationLoss.getOutput()->row(y * HEIGHT + x);
            // blue: 72, 133, 232
            // red: 255, 130, 130
            // green: 109, 209, 129
            pixels[index + 0] = sqrt(72 * 72 * pix(0) + 255 * 255 * pix(1) + 109 * 109 * pix(2));
            pixels[index + 1] = sqrt(133 * 133 * pix(0) + 130 * 130 * pix(1) + 209 * 209 * pix(2));
            pixels[index + 2] = sqrt(232 * 232 * pix(0) + 130 * 130 * pix(1) + 129 * 129 * pix(2));
            pixels[index + 3] = 255; // Alpha channel (opacity: 255 = fully opaque)
            // pixels[4 * (y * WIDTH + x) + gridPredictions(HEIGHT * y + x, 0)] = 255;
        }
    }
    ImageGenerator gen = ImageGenerator();
    gen.createImage(pixels, "visualizations/lr1dr0m0.png", WIDTH, HEIGHT);

    return 0;
}