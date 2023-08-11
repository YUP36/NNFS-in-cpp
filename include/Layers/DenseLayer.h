#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <iostream>
#include <Eigen/Dense>

class DenseLayer {

    public:
        DenseLayer(int numInputs, int numNeurons, double l1w = 0.0, double l1b = 0.0, double l2w = 0.0, double l2b = 0.0);
        friend std::ostream& operator<<(std::ostream& os, const DenseLayer& layer);
        
        Eigen::MatrixXd* getWeights() const;
        void setWeights(Eigen::MatrixXd* newWeights);
        void updateWeights(Eigen::MatrixXd* weightsUpdate);
        Eigen::RowVectorXd* getBiases() const;
        void setBiases(Eigen::RowVectorXd* newBiases);
        void updateBiases(Eigen::RowVectorXd* biasesUpdate);
        void forward(Eigen::MatrixXd* in);
        Eigen::MatrixXd* getOutput() const;

        void backward(Eigen::MatrixXd* dvalues);
        Eigen::MatrixXd* getDinputs() const;
        Eigen::MatrixXd* getDweights() const;
        Eigen::RowVectorXd* getDbiases() const;

        Eigen::MatrixXd* getWeightMomentum() const;
        void setWeightMomentum(Eigen::MatrixXd* newWeightMomentum);
        Eigen::RowVectorXd* getBiasMomentum() const;
        void setBiasMomentum(Eigen::RowVectorXd* newBiasMomentum);

        Eigen::MatrixXd* getWeightCache() const;
        void setWeightCache(Eigen::MatrixXd* update);
        void updateWeightCache(Eigen::MatrixXd* update);
        Eigen::RowVectorXd* getBiasCache() const;
        void setBiasCache(Eigen::RowVectorXd* update);
        void updateBiasCache(Eigen::RowVectorXd* update);

        double getLambdaL1Weight() const;
        double getLambdaL1Bias() const;
        double getLambdaL2Weight() const;
        double getLambdaL2Bias() const;
        
    private:
        Eigen::MatrixXd* input;
        Eigen::MatrixXd* output;

        Eigen::MatrixXd* weights;
        Eigen::RowVectorXd* biases;

        Eigen::MatrixXd* dinputs;
        Eigen::MatrixXd* dweights;
        Eigen::RowVectorXd* dbiases;

        Eigen::MatrixXd* weightMomentum;
        Eigen::RowVectorXd* biasMomentum;

        Eigen::MatrixXd* weightCache;
        Eigen::RowVectorXd* biasCache;

        double lambdaL1Weight;
        double lambdaL1Bias;
        double lambdaL2Weight;
        double lambdaL2Bias;
};

#endif