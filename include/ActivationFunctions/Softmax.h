#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>
#include "../ModelWrappers/Layer.h"
#include "../ModelWrappers/Activation.h"

class Softmax : public Activation {

    public:
        Softmax();
        std::string getName() const override;
        
        void forward(Eigen::MatrixXd* input) override;
        Eigen::MatrixXd* getOutput() const override;
        Eigen::MatrixXd getPredictions() const override;
        
        void backward(Eigen::MatrixXd* dvalues) override;
        Eigen::MatrixXd* getDinputs() const override;
        
    private:
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* dinputs;

};

#endif