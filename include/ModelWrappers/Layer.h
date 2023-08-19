#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

class Layer {

    public:
        Layer();
        virtual std::string getName() const;

        virtual void forward(Eigen::MatrixXd* in);
        virtual Eigen::MatrixXd* getOutput() const;

        virtual void backward(Eigen::MatrixXd* dvalues);
        virtual Eigen::MatrixXd* getDinputs() const;

};

#endif