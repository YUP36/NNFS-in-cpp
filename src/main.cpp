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
#include "../include/LossFunctions/BinaryCrossEntropy.h"
#include "../include/LossFunctions/MeanAbsoluteError.h"
#include "../include/LossFunctions/MeanSquaredError.h"

#include "../include/Optimizers/SGD.h"
#include "../include/Optimizers/Adagrad.h"
#include "../include/Optimizers/RMSProp.h"
#include "../include/Optimizers/Adam.h"

#include "../include/AccuracyCalculations/RegressionAccuracy.h"
#include "../include/AccuracyCalculations/CategoricalAccuracy.h"

#include "../include/ModelWrappers/SoftmaxCategoricalCrossEntropy.h"
#include "../include/ModelWrappers/Model.h"
#include "../include/ModelWrappers/Layer.h"
#include "../include/ModelWrappers/Optimizer.h"
#include "../include/ModelWrappers/Loss.h"
#include "../include/ModelWrappers/Accuracy.h"

using namespace std;
using Eigen::MatrixXd;
using namespace std::chrono;

// auto start = high_resolution_clock::now();

// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << duration.count() << std::endl;

int main() {
    srand(1);

    // Sine dataset = Sine();
    Spiral dataset = Spiral(100, 2);
    MatrixXd X = dataset.getX();
    MatrixXd Y = dataset.getY();

    // Sine validationData = Sine();
    Spiral validationData = Spiral(100, 2);
    MatrixXd XValidation = validationData.getX();
    MatrixXd YValidation = validationData.getY();

    // Layer* layer1 = new Dense(1, 64);
    // Layer* activation1 = new ReLu();
    // Layer* layer2 = new Dense(64, 64);
    // Layer* activation2 = new ReLu();
    // Layer* layer3 = new Dense(64, 1);
    // Layer* activation3 = new Linear();

    // Layer* layer1 = new Dense(2, 64, 5e-4, 5e-4);
    // Layer* activation1 = new ReLu();
    // Layer* layer2 = new Dense(64, 1);
    // Layer* activation2 = new Sigmoid();

    Layer* layer1 = new Dense(2, 64, 0.0, 0.0, 5e-4, 5e-4);
    Layer* activation1 = new ReLu();
    // Layer* dropout = new Dropout(0.1);
    Layer* layer2 = new Dense(64, 5);
    Layer* activation2 = new Softmax();
    
    // Loss* loss = new MeanSquaredError();
    // Loss* loss = new BinaryCrossEntropy();
    Loss* loss = new CategoricalCrossEntropy();
    
    Optimizer* optimizer = new Adam(0.001, 5e-5);
    
    // Accuracy* accuracy = new RegressionAccuracy();
    Accuracy* accuracy = new CategoricalAccuracy();

    Model m = Model();    

    m.add(layer1);
    m.add(activation1);
    // m.add(dropout);
    m.add(layer2);
    m.add(activation2);
    // m.add(layer3);
    // m.add(activation3);
    m.set(loss, optimizer, accuracy);

    m.finalize();

    m.train(&X, &Y, 10000, 100, &XValidation, &YValidation);
    
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