#include <iostream>
#include <vector>
#include <random>

#include "../include/DataGeneration/lodepng.h"
#include "../include/DataGeneration/ImageGenerator.h"
#include "../include/DataGeneration/DataGenerator.h"
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
using Eigen::RowVectorXd;
using namespace std::chrono;

// auto start = high_resolution_clock::now();

// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << duration.count() << std::endl;

int main() {
    srand(1);
    auto rng = std::default_random_engine {};

    string path = "data/FashionMnistImages/";
    DataGenerator data = DataGenerator();

    data.loadMnistDataset(path + "train/");
    MatrixXd XTrain = data.getX();
    MatrixXd YTrain = data.getY();

    data.loadMnistDataset(path + "test/");
    MatrixXd XValidation = data.getX();
    MatrixXd YValidation = data.getY();

    vector<int> keys;
    for(int i = 0; i < YTrain.rows(); i++) {
        keys.push_back(i);
    }
    std::shuffle(keys.begin(), keys.end(), rng);

    XTrain = XTrain(keys, Eigen::placeholders::all);
    YTrain = YTrain(keys, Eigen::placeholders::all);

    XTrain = (XTrain.array() - 127.5) / 127.5;
    XValidation = (XValidation.array() - 127.5) / 127.5;

    Layer* layer1 = new Dense(XTrain.cols(), 64);
    Layer* activation1 = new ReLu();
    Layer* layer2 = new Dense(64, 64);
    Layer* activation2 = new ReLu();
    Layer* layer3 = new Dense(64, 10);
    Layer* activation3 = new Softmax();
    
    Loss* loss = new CategoricalCrossEntropy();
    Optimizer* optimizer = new Adam(0.001, 5e-5);
    Accuracy* accuracy = new CategoricalAccuracy();

    Model m = Model();    

    m.add(layer1);
    m.add(activation1);
    m.add(layer2);
    m.add(activation2);
    m.add(layer3);
    m.add(activation3);
    m.set(loss, optimizer, accuracy);

    m.finalize();

    m.train(&XTrain, &YTrain, 5, 100, 128, &XValidation, &YValidation);
    m.evaluate(&XValidation, &YValidation, 128);

    MatrixXd slice = XValidation.middleRows(0, 5);
    // cout << m.predict(&slice, 10, 0) << endl;

    ImageGenerator gen = ImageGenerator();
    vector<string> names = {"pants.png", "tshirt.png"};
    for(int i = 0; i < names.size(); i++) {
        vector<unsigned char> rawImage = gen.decodeImage(("data/fashionImages/" + names[i]).c_str());
        vector<unsigned char> rescaledImage;
        vector<double> grayScaleImage;

        int dim = (int) sqrt(rawImage.size()) / 2;
        for(int row = 0; row < dim - dim / 28; row += dim / 28) {
            for(int col = 0; col < dim - dim / 28; col += dim / 28) {
                double grayInvertedPixel = 255 - (0.299 * rawImage[(row * dim + col) * 4] 
                                            + 0.587 * rawImage[(row * dim + col + 1) * 4] 
                                            + 0.114 * rawImage[(row * dim + col + 2) * 4]);
                grayScaleImage.push_back(grayInvertedPixel);
                rescaledImage.push_back((unsigned char) grayInvertedPixel);
                rescaledImage.push_back((unsigned char) grayInvertedPixel);
                rescaledImage.push_back((unsigned char) grayInvertedPixel);
                rescaledImage.push_back((unsigned char) 255);
            }
        }

        lodepng::encode(("data/fashionImages/rescaled_" + names[i]).c_str(), rescaledImage, 28, 28);
        MatrixXd sample = Eigen::Map<RowVectorXd>(grayScaleImage.data(), 1, grayScaleImage.size());
        sample = (sample.array() - 127.5) / 127.5;

        cout << m.predict(&sample, 10, 0) << endl;
    }
        

    // unsigned error = lodepng::encode("data/", pixels, WIDTH, HEIGHT);
    
    // vector<MatrixXd> weights = m.getWeights();
    // vector<RowVectorXd> biases = m.getBiases();

    // // NEW MODEL!
    // Layer* m2layer1 = new Dense(XTrain.cols(), 64);
    // Layer* m2activation1 = new ReLu();
    // Layer* m2layer2 = new Dense(64, 64);
    // Layer* m2activation2 = new ReLu();
    // Layer* m2layer3 = new Dense(64, 10);
    // Layer* m2activation3 = new Softmax();
    
    // Loss* m2loss = new CategoricalCrossEntropy();
    // Optimizer* m2optimizer = new Adam(0.001, 5e-5);
    // Accuracy* m2accuracy = new CategoricalAccuracy();

    // Model m2 = Model();    

    // m2.add(m2layer1);
    // m2.add(m2activation1);
    // m2.add(m2layer2);
    // m2.add(m2activation2);
    // m2.add(m2layer3);
    // m2.add(m2activation3);
    // m2.set(m2loss, m2optimizer, m2accuracy);

    // m2.finalize();

    // m2.setParameters(weights, biases);
    // m2.evaluate(&XValidation, &YValidation, 128);

    return 0;
}