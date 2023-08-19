#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <Eigen/Dense>
#include "../ModelWrappers/Loss.h"

class CategoricalCrossEntropy : public Loss {

    public:
        CategoricalCrossEntropy();
        Eigen::VectorXd forward(Eigen::MatrixXd* yPredicted, Eigen::MatrixXd* yTrue) override;
        void backward(Eigen::MatrixXd* yPredicted, Eigen::VectorXi* yTrue);
        Eigen::MatrixXd* getDinputs() const;

    private:
        Eigen::MatrixXd* dinputs;
};

#endif