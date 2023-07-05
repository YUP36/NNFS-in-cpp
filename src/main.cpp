#include <iostream>
#include "../include/DenseLayer.h"
// #include <Eigen/Dense>
 
// using Eigen::MatrixXd;

int main() {
    DenseLayer layer(3, 5);
    layer.printLayer();
    return 0;
}