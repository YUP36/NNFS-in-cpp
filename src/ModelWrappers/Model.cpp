#include "../../include/ModelWrappers/Model.h"

#include "../../include/Layers/Dropout.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <iostream>
using namespace std;

Model::Model() {
    activationLoss = nullptr;
}

void Model::add(Layer* layerPointer) {
    layers.push_back(layerPointer);
}

void Model::set(Loss* lo, Optimizer* op, Accuracy* acc) {
    loss = lo;
    optimizer = op;
    accuracy = acc;
}

void Model::finalize() {
    numLayers = layers.size();
    for(int i = 0; i < numLayers; i++) {
        if(layers[i]->getName() == "Dense") {
            trainableLayers.push_back(dynamic_cast<Dense*>(layers[i]));
        }
    }
    outputLayer = dynamic_cast<Activation*>(layers[numLayers - 1]);
    if(outputLayer->getName() == "Softmax" && loss->getName() == "CategoricalCrossEntropy") {
        activationLoss = new SoftmaxCategoricalCrossEntropy();
    }
}

void Model::train(MatrixXd* X, MatrixXd* Y, int epochs, int printEvery, MatrixXd* XValidation, MatrixXd* YValidation) {
    accuracy->initialize(Y);

    for(int epoch = 0; epoch < epochs; epoch++) {

        // FORWARD PASS
        MatrixXd output = forward(X, true);

        // LOSS CALCULATION
        double dataLoss = loss->calculate(&output, Y);
        double regularizationLoss = 0;
        for(int i = 0; i < trainableLayers.size(); i++) {
            regularizationLoss += loss->calculateRegularizationLoss(trainableLayers[i]);
        } 
        double totalLoss = dataLoss + regularizationLoss;

        // ACCURACY CALCULATION
        MatrixXd predictions = outputLayer->getPredictions();
        double accuracyScore = accuracy->calculateAccuracy(&predictions, Y);

        // BACKPROPAGATION
        backward(&output, Y);

        // PARAMETER UPDATE
        optimizer->decay();
        for(int i = 0; i < trainableLayers.size(); i++) {
            optimizer->updateParameters(trainableLayers[i]);
        } 
        optimizer->incrementIteration();

        // PRINT EPOCH
        if(epoch % printEvery == 0) {
            cout << "Epoch: " << epoch << "\t";
            cout << "Loss: " << totalLoss << "\t";
            cout << "Data loss: " << dataLoss << "\t";
            cout << "Regularization loss: " << regularizationLoss << "\t";
            cout << "Accuracy: " << accuracyScore << endl;
        }
    }

    // VALIDATION
    if(YValidation && XValidation) {
        // FORWARD PASS
        MatrixXd output = forward(XValidation, false);
        // LOSS CALCULATION
        double totalLoss = loss->calculate(&output, YValidation);
        // ACCURACY CALCULATION
        MatrixXd predictions = outputLayer->getPredictions();
        double accuracyScore = accuracy->calculateAccuracy(&predictions, YValidation);
        // PRINT VALIDATION RESULTS
        cout << endl << "VALIDATION\t";
        cout << "Loss: " << totalLoss << "\t";
        cout << "Accuracy: " << accuracyScore << endl;
    }
}

MatrixXd Model::forward(MatrixXd* in, bool training) {
    layers[0]->forward(in);
    for(int i = 1; i < numLayers; i++) {
        if(layers[i]->getName() == "Dropout") {
            (dynamic_cast<Dropout*>(layers[i]))->forward(layers[i - 1]->getOutput(), training);
        } else {
            layers[i]->forward(layers[i - 1]->getOutput());
        }
    }
    return *(outputLayer->getOutput());
}

void Model::backward(MatrixXd* output, MatrixXd* yTrue) {
    if(activationLoss) {
        activationLoss->backward(output, yTrue);
        if(numLayers >= 2) layers[numLayers - 2]->backward(activationLoss->getDinputs());
    } else {
        loss->backward(output, yTrue);
        outputLayer->backward(loss->getDinputs());
        if(numLayers >= 2) layers[numLayers - 2]->backward(outputLayer->getDinputs());
    }

    for(int i = numLayers - 3; i >= 0; i--) {
        layers[i]->backward(layers[i + 1]->getDinputs());
    }
}