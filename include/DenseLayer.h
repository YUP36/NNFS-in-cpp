#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <iostream>
#include <Eigen/Dense>

class DenseLayer {

    public:
        DenseLayer(int numInputs, int numNeurons);
        friend std::ostream& operator<<(std::ostream& os, const DenseLayer& layer);
        void printLayer() const;
        Eigen::MatrixXd getWeights() const;
        Eigen::RowVectorXd getBiases() const;
        void forward(Eigen::MatrixXd inputs);
        Eigen::MatrixXd getOutput() const;
        
    private:
        Eigen::MatrixXd weights;
        Eigen::RowVectorXd biases;
        Eigen::MatrixXd output;

};

#endif