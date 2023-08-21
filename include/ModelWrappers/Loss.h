#ifndef LOSS_H
#define LOSS_H

#include "../Layers/Dense.h"
#include "NamedWrapper.h"

#include <Eigen/Dense>

class Loss : public NamedWrapper {

    public:
        Loss();
        std::string getName() const override;

        double calculate(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        virtual void forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        virtual Eigen::VectorXd* getOutput();
        
        virtual void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        virtual Eigen::MatrixXd* getDinputs() const;
        
        double calculateRegularizationLoss(Dense* layer);

};

#endif