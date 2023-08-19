#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <Eigen/Dense>

#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"

class Model {

    public:
        Model();
        void add(Layer* layerPointer);
        void set(Loss* loss, Optimizer* optimizer);
        void finalize();
        void train(Eigen::MatrixXd* X, Eigen::MatrixXd* Y, int epochs, int printEvery);
        Eigen::MatrixXd forward(Eigen::MatrixXd* in);

    private:
        std::vector<Layer*> layers;
        std::vector<Layer*> trainableLayers;

        Loss* loss;
        Optimizer* optimizer;
        Layer* outputLayer;

};

#endif