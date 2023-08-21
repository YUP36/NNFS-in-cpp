#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

#include <Eigen/Dense>
#include "../ModelWrappers/Loss.h"

class MeanSquaredError : public Loss {

    public:
        MeanSquaredError();
        std::string getName() const override;

        void forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        virtual Eigen::VectorXd* getOutput() override;
        
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        Eigen::MatrixXd* getDinputs() const override;

    private:
        Eigen::VectorXd* output;
        Eigen::MatrixXd* dinputs;
};

#endif