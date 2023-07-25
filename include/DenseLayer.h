#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <iostream>
#include <Eigen/Dense>

class DenseLayer {

    public:
        DenseLayer(int numInputs, int numNeurons);
        friend std::ostream& operator<<(std::ostream& os, const DenseLayer& layer);
        void printLayer() const;
        
        Eigen::MatrixXd* getWeights() const;
        void setWeights(Eigen::MatrixXd newWeights);
        Eigen::RowVectorXd* getBiases() const;
        void setBiases(Eigen::RowVectorXd newBiases);

        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput() const;

        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;
        Eigen::MatrixXd* getDweights() const;
        Eigen::RowVectorXd* getDbiases() const;
        
    private:
        Eigen::MatrixXd* input;
        Eigen::MatrixXd* output;

        Eigen::MatrixXd* weights;
        Eigen::RowVectorXd* biases;

        Eigen::MatrixXd* dinputs;
        Eigen::MatrixXd* dweights;
        Eigen::RowVectorXd* dbiases;

};

#endif