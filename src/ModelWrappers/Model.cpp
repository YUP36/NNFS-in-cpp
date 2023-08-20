#include "../../include/ModelWrappers/Model.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <iostream>
using namespace std;

Model::Model() {}

void Model::add(Layer* layerPointer) {
    layers.push_back(layerPointer);
}

void Model::set(Loss* lo, Optimizer* op, Accuracy* acc) {
    loss = lo;
    optimizer = op;
    accuracy = acc;
}

void Model::finalize() {
    for(int i = 0; i < layers.size(); i++) {
        if(layers[i]->getName() == "Dense") {
            trainableLayers.push_back(dynamic_cast<Dense*>(layers[i]));
        }
    }
    outputLayer = dynamic_cast<Activation*>(layers[layers.size() - 1]);
}

void Model::train(MatrixXd* X, MatrixXd* Y, int epochs, int printEvery) {
    accuracy->initialize(Y);

    for(int epoch = 0; epoch < epochs; epoch++) {
        MatrixXd output = forward(X);

        double dataLoss = loss->calculate(&output, Y);
        double regularizationLoss = 0;
        for(int i = 0; i < trainableLayers.size(); i++) {
            regularizationLoss += loss->calculateRegularizationLoss(trainableLayers[i]);
        } 
        double loss = dataLoss + regularizationLoss;

        MatrixXd predictions = outputLayer->getPredictions();
        double accuracyScore = accuracy->calculateAccuracy(&predictions, Y);

        cout << "Epoch: " << epoch << "\t";
        cout << "Loss: " << loss << "\t";
        cout << "Data loss: " << dataLoss << "\t";
        cout << "Regularization loss: " << regularizationLoss << "\t";
        cout << "Accuracy: " << accuracyScore << endl;
    }
}

MatrixXd Model::forward(MatrixXd* in) {
    layers[0]->forward(in);
    for(int i = 1; i < layers.size(); i++) {
        // cout << layers[i]->getName() << endl;
        layers[i]->forward(layers[i - 1]->getOutput());
    }
    return *(outputLayer->getOutput());
}

