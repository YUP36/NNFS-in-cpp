#ifndef LOSSCATEGORICALCROSSENTROPY_H
#define LOSSCATEGORICALCROSSENTROPY_H

#include <Eigen/Dense>
#include "Loss.h"

class LossCategoricalCrossEntropy : public Loss {

    public:
        LossCategoricalCrossEntropy();
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::VectorXi* yTrue) override;
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::VectorXi* yTrue);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* dinputs;
};

#endif