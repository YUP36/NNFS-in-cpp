#ifndef LAYER_H
#define LAYER_H

#include "../ModelWrappers/NamedWrapper.h"

#include <Eigen/Dense>

class Layer : public NamedWrapper {

    public:
        Layer();
        std::string getName() const override;

        virtual void forward(Eigen::MatrixXd* in);
        virtual Eigen::MatrixXd* getOutput() const;

        virtual void backward(Eigen::MatrixXd* dvalues);
        virtual Eigen::MatrixXd* getDinputs() const;

};

#endif