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

        void train(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, int epochs, int printEvery, int batchSize = 0, Eigen::MatrixXd* XValidation = nullptr, Eigen::MatrixXd* YValidation = nullptr);
        void evaluate(Eigen::MatrixXd* XValidation, Eigen::MatrixXd* YValidation, int batchSize = 0);
        Eigen::MatrixXd predict(Eigen::MatrixXd* X, int numLabels, int batchSize = 0);

        std::vector<Eigen::MatrixXd> getWeights();
        std::vector<Eigen::RowVectorXd> getBiases();
        void setParameters(std::vector<Eigen::MatrixXd> weights, std::vector<Eigen::RowVectorXd> biases);

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