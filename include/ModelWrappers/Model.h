#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <Eigen/Dense>

#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Accuracy.h"
#include "Activation.h"
#include "../Layers/Dense.h"
#include "SoftmaxCategoricalCrossEntropy.h"

class Model {

    public:
        Model();
        void add(Layer* layerPointer);
        void set(Loss* loss, Optimizer* optimizer, Accuracy* acc);
        void finalize();
        void train(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, int epochs, int printEvery, Eigen::MatrixXd* XValidation = nullptr, Eigen::MatrixXd* YValidation = nullptr);
        Eigen::MatrixXd forward(Eigen::MatrixXd* in, bool training);
        void backward(Eigen::MatrixXd* output, Eigen::MatrixXd* yTrue);

    private:
        std::vector<Layer*> layers;
        double numLayers;
        std::vector<Dense*> trainableLayers;

        Loss* loss;
        Optimizer* optimizer;
        Accuracy* accuracy;

        Activation* outputLayer;
        SoftmaxCategoricalCrossEntropy* activationLoss;

};

#endif