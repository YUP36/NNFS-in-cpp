#ifndef BINARYCROSSENTROPY_H
#define BINARYCROSSENTROPY_H

#include <Eigen/Dense>
#include "../ModelWrappers/Loss.h"

class BinaryCrossEntropy : public Loss {

    public:
        BinaryCrossEntropy();
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* dinputs;

};

#endif