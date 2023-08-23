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

void Model::train(MatrixXd* X, MatrixXd* Y, int epochs, int printEvery, int batchSize, MatrixXd* XValidation, MatrixXd* YValidation) {
    accuracy->initialize(Y);
    
    int numSamples = X->rows();
    int numTrainingSteps;
    if(batchSize == 0) {
        numTrainingSteps = 1;
        batchSize = numSamples;
    } else {
        numTrainingSteps = (numSamples / batchSize) + ((numSamples % batchSize) > 0);
    }

    for(int epoch = 0; epoch < epochs; epoch++) {

        loss->newPass();
        accuracy->newPass();
        cout << "Epoch: " << epoch << endl;

        double dataLoss, regularizationLoss, totalLoss;
        for(int step = 0; step < numTrainingSteps; step++) {
            // SLICE BATCH
            int actualBatchSize = (step < numTrainingSteps - 1) ? batchSize : numSamples - (step * batchSize);
            MatrixXd XBatch = X->middleRows(step * batchSize, actualBatchSize);
            MatrixXd YBatch = Y->middleRows(step * batchSize, actualBatchSize);

            // FORWARD PASS
            MatrixXd output = forward(&XBatch, true);

            // LOSS CALCULATION
            dataLoss = loss->calculate(&output, &YBatch);
            regularizationLoss = 0;
            for(int i = 0; i < trainableLayers.size(); i++) {
                regularizationLoss += loss->calculateRegularizationLoss(trainableLayers[i]);
            } 
            totalLoss = dataLoss + regularizationLoss;

            // ACCURACY CALCULATION
            MatrixXd predictions = outputLayer->getPredictions();
            double accuracyScore = accuracy->calculateAccuracy(&predictions, &YBatch);

            // BACKPROPAGATION
            backward(&output, &YBatch);

            // PARAMETER UPDATE
            optimizer->decay();
            for(int i = 0; i < trainableLayers.size(); i++) {
                optimizer->updateParameters(trainableLayers[i]);
            } 
            optimizer->incrementIteration();

            // PRINT STEP
            if((step % printEvery == 0) || (step == numTrainingSteps - 1)) {
                cout << "\tStep: " << step << "\t";
                cout << "Loss: " << totalLoss << "\t";
                cout << "Data loss: " << dataLoss << "\t";
                cout << "Regularization loss: " << regularizationLoss << "\t";
                cout << "Accuracy: " << accuracyScore << "\t";
                cout << "Learning rate: " << optimizer->getLearningRate() << endl;
            }
        }

        cout << "Completed epoch " << epoch << "\t";
        cout << "Loss: " << loss->getAverageDataLoss() + regularizationLoss << "\t";
        cout << "Data loss: " << loss->getAverageDataLoss() << "\t";
        cout << "Regularization loss: " << regularizationLoss << "\t";
        cout << "Accuracy: " << accuracy->getAverageAccuracy() << "\t";
        cout << "Learning rate: " << optimizer->getLearningRate() << endl << endl;
    }

    // VALIDATION
    if(YValidation && XValidation) {
        loss->newPass();
        accuracy->newPass();

        int numSamples = XValidation->rows();
        int numTrainingSteps;
        if(batchSize == 0) {
            numTrainingSteps = 1;
            batchSize = numSamples;
        } else {
            numTrainingSteps = (numSamples / batchSize) + ((numSamples % batchSize) > 0);
        }

        for(int step = 0; step < numTrainingSteps; step++) {
            // SLICE BATCH
            int actualBatchSize = (step < numTrainingSteps - 1) ? batchSize : numSamples - (step * batchSize);
            MatrixXd XBatch = X->middleRows(step * batchSize, actualBatchSize);
            MatrixXd YBatch = Y->middleRows(step * batchSize, actualBatchSize);

            // FORWARD PASS
            MatrixXd output = forward(&XBatch, false);

            // LOSS CALCULATION
            loss->calculate(&output, &YBatch);

            // ACCURACY CALCULATION
            MatrixXd predictions = outputLayer->getPredictions();
            accuracy->calculateAccuracy(&predictions, &YBatch);
        }

        // PRINT VALIDATION RESULTS
        cout << "VALIDATION\t";
        cout << "Loss: " << loss->getAverageDataLoss() << "\t";
        cout << "Accuracy: " << accuracy->getAverageAccuracy() << endl;
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