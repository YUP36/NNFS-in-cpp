#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <Eigen/Dense>
#include "../ModelWrappers/Layer.h"

class Dense : public Layer {

    public:
        Dense(int numInputs, int numNeurons, double l1w = 0.0, double l1b = 0.0, double l2w = 0.0, double l2b = 0.0);
        std::string getName() const override;
        friend std::ostream& operator<<(std::ostream& os, const Dense& layer);
        
        Eigen::MatrixXd* getWeights() const;
        void setWeights(Eigen::MatrixXd* newWeights);
        void updateWeights(Eigen::MatrixXd* weightsUpdate);
        Eigen::RowVectorXd* getBiases() const;
        void setBiases(Eigen::RowVectorXd* newBiases);
        void updateBiases(Eigen::RowVectorXd* biasesUpdate);
        void forward(Eigen::MatrixXd* in) override;
        Eigen::MatrixXd* getOutput() const override;

        void backward(Eigen::MatrixXd* dvalues) override;
        Eigen::MatrixXd* getDinputs() const override;
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