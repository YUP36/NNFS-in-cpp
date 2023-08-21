#include "../../include/ModelWrappers/Layer.h"

using Eigen::MatrixXd;

Layer::Layer() {}

std::string Layer::getName() const {
    return "Layer";
}

void Layer::forward(MatrixXd* in) {
    return;
}

MatrixXd* Layer::getOutput() const {
    return new MatrixXd();
}

void Layer::backward(MatrixXd* dvalues) {
    return;
}

MatrixXd* Layer::getDinputs() const {
    return new MatrixXd();
}