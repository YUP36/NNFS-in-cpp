#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

class DenseLayer {

    private:
        MatrixXd weights;
        RowVectorXd biases;
        MatrixXd output;

    public:
        DenseLayer(int numInputs, int numNeurons);
        void printLayer() const;
        MatrixXd getWeights() const;
        RowVectorXd getBiases() const;
        void forward(MatrixXd inputs);
        MatrixXd getOutput() const;
};

#endif