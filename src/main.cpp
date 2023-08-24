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

void mnistExample();
void cceExample();
void bceExample();
void mseExample();

int main() {
    srand(1);

    // mnistExample();
    // cceExample();
    // bceExample();
    // mseExample();
        
    return 0;
}

void mnistExample() {

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
    auto rng = std::default_random_engine {};
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

    // MatrixXd slice = XValidation.middleRows(0, 5);
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
}

void cceExample() {
    const int NUMSAMPLES = 100;
    const int NUMLABELS = 3;
    const int WIDTH = 1000;
    const int HEIGHT = 1000;
    const int EPOCHS = 10000;
    const int PRINTEVERY = 100;
    const int BATCHSIZE = 0;

    Spiral dataset = Spiral(NUMSAMPLES, NUMLABELS);
    MatrixXd X = dataset.getX();
    MatrixXd Y = dataset.getY();

    Spiral validationData = Spiral(NUMSAMPLES, NUMLABELS);
    MatrixXd XValidation = validationData.getX();
    MatrixXd YValidation = validationData.getY();

    Layer* layer1 = new Dense(2, 64, 0.0, 0.0, 5e-4, 5e-4);
    Layer* activation1 = new ReLu();
    // Layer* dropout = new Dropout(0.1);
    Layer* layer2 = new Dense(64, NUMLABELS);
    Layer* activation2 = new Softmax();
    
    Loss* loss = new CategoricalCrossEntropy();
    
    Optimizer* optimizer = new Adam(0.02, 5e-7);
    
    Accuracy* accuracy = new CategoricalAccuracy();

    Model m = Model();    

    m.add(layer1);
    m.add(activation1);
    // m.add(dropout);
    m.add(layer2);
    m.add(activation2);
    m.set(loss, optimizer, accuracy);

    m.finalize();

    m.train(&X, &Y, EPOCHS, PRINTEVERY, BATCHSIZE, &XValidation, &YValidation);

    ImageGenerator gen = ImageGenerator();
    gen.visualize2DClassifier(&m, "data/visualizations/examples/ex1.png", NUMLABELS, false, WIDTH, HEIGHT);
}

void bceExample() {
    const int NUMSAMPLES = 100;
    const int NUMLABELS = 1;
    const int WIDTH = 1000;
    const int HEIGHT = 1000;
    const int EPOCHS = 10000;
    const int PRINTEVERY = 100;
    const int BATCHSIZE = 0;

    Spiral dataset = Spiral(NUMSAMPLES, 2);
    MatrixXd X = dataset.getX();
    MatrixXd Y = dataset.getY();

    Spiral validationData = Spiral(NUMSAMPLES, 2);
    MatrixXd XValidation = validationData.getX();
    MatrixXd YValidation = validationData.getY();

    Layer* layer1 = new Dense(2, 64, 5e-4, 5e-4);
    Layer* activation1 = new ReLu();
    Layer* layer2 = new Dense(64, 1);
    Layer* activation2 = new Sigmoid();

    Loss* loss = new BinaryCrossEntropy();
    
    Optimizer* optimizer = new Adam(0.001, 5e-5);
    
    Accuracy* accuracy = new CategoricalAccuracy();

    Model m = Model();    

    m.add(layer1);
    m.add(activation1);
    m.add(layer2);
    m.add(activation2);
    m.set(loss, optimizer, accuracy);

    m.finalize();

    m.train(&X, &Y, EPOCHS, PRINTEVERY, BATCHSIZE, &XValidation, &YValidation);

    ImageGenerator gen = ImageGenerator();
    gen.visualize2DClassifier(&m, "data/visualizations/examples/ex2.png", NUMLABELS, true, WIDTH, HEIGHT);
}

void mseExample() {
    const int NUMSAMPLES = 1000;
    const int EPOCHS = 10000;
    const int PRINTEVERY = 100;
    const int BATCHSIZE = 0;

    Sine dataset = Sine(NUMSAMPLES);
    MatrixXd X = dataset.getX();
    MatrixXd Y = dataset.getY();

    Sine validationData = Sine(NUMSAMPLES);
    MatrixXd XValidation = validationData.getX();
    MatrixXd YValidation = validationData.getY();

    Layer* layer1 = new Dense(1, 64);
    Layer* activation1 = new ReLu();
    Layer* layer2 = new Dense(64, 64);
    Layer* activation2 = new ReLu();
    Layer* layer3 = new Dense(64, 1);
    Layer* activation3 = new Linear();
    
    Loss* loss = new MeanSquaredError();
    
    Optimizer* optimizer = new Adam(0.001, 5e-5);
    
    Accuracy* accuracy = new RegressionAccuracy();

    Model m = Model();    

    m.add(layer1);
    m.add(activation1);
    m.add(layer2);
    m.add(activation2);
    m.add(layer3);
    m.add(activation3);
    m.set(loss, optimizer, accuracy);

    m.finalize();

    m.train(&X, &Y, EPOCHS, PRINTEVERY, BATCHSIZE, &XValidation, &YValidation);
}
