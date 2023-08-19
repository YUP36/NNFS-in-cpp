#include "../../include/ModelWrappers/Model.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Model::Model() {}

void Model::add(Layer* layerPointer) {
    layers.push_back(layerPointer);
}

void Model::set(Loss* lo, Optimizer* op) {
    loss = lo;
    optimizer = op;
}

void Model::finalize() {
    for(int i = 0; i < layers.size(); i++) {
        if(layers[i]->getName() == "Dense") {
            trainableLayers.push_back(layers[i]);
        }
    }
    outputLayer = layers[layers.size() - 1];
}

void Model::train(MatrixXd* X, MatrixXd* Y, int epochs, int printEvery) {
    for(int epoch = 0; epoch < epochs; epoch++) {
        MatrixXd output = forward(X);
    }
}

MatrixXd Model::forward(MatrixXd* in) {
    layers[0]->forward(in);
    for(int i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 0]->getOutput());
    }
    return *(outputLayer->getOutput());
}

