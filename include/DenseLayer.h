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
        Eigen::MatrixXd getOutput() const;
        void forward(Eigen::MatrixXd inputs);
        void backward(Eigen::MatrixXd dvalues);
        
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd output;
        Eigen::MatrixXd weights;
        Eigen::RowVectorXd biases;
        Eigen::MatrixXd dinputs;
        Eigen::MatrixXd dweights;
        Eigen::RowVectorXd dbiases;

};

#endif