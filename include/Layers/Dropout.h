#ifndef DROPOUT_H
#define DROPOUT_H

#include <Eigen/Dense>
#include "../ModelWrappers/Layer.h"

class Dropout : public Layer {
    // dropout rate only has accuracy to 5 decimal places

    public:
        Dropout(double r);
        std::string getName() const override;

        void forward(Eigen::MatrixXd* in) override;
        void forward(Eigen::MatrixXd* in, bool training);
        Eigen::MatrixXd* getOutput() const override;

        void backward(Eigen::MatrixXd* dvalues) override;
        Eigen::MatrixXd* getDinputs() const override;

    private:
        double dropoutRate;
        double cutoff;
        int PRECISION;
        Eigen::MatrixXd* output;
        Eigen::MatrixXd* mask;
        Eigen::MatrixXd* dinputs;

};

#endif