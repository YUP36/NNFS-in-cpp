#ifndef RELU_H
#define RELU_H

#include <Eigen/Dense>
#include "../ModelWrappers/Layer.h"
#include "../ModelWrappers/Activation.h"

class ReLu : public Layer, public Activation {

    public:
        ReLu();
        std::string getName() const override;

        void forward(Eigen::MatrixXd* in) override;
        Eigen::MatrixXd* getOutput() const override;
        Eigen::MatrixXd getPredictions() const override;

        void backward(Eigen::MatrixXd* dvalues) override;
        Eigen::MatrixXd* getDinputs() const override;

    private:
        Eigen::MatrixXd* input;
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif