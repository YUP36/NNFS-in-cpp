#include "../../include/DataGeneration/Sine.h"

#include <cmath>

using Eigen::VectorXd;

Sine::Sine(int numSamples) {
    X = VectorXd::LinSpaced(numSamples, 0, 1);
    Y = X.unaryExpr([](double x){return sin(2 * M_PI * x);});
}

std::ostream& operator<<(std::ostream& os, const Sine& dataset) {
    os << "X (Inputs):\n" << dataset.getX() << std::endl << "Y (Labels):\n" << dataset.getY() << std::endl;
    return os;
}

VectorXd Sine::getX() const {
    return X;
}

VectorXd Sine::getY() const {
    return Y;
}

