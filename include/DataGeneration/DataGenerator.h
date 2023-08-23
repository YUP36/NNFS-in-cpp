#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include <string>
#include <Eigen/Dense>

class DataGenerator {
    
    public:
        DataGenerator();
        void loadMnistDataset(std::string path);
        Eigen::MatrixXd getX();
        Eigen::MatrixXd getY();

    private:
        Eigen::MatrixXd X;
        Eigen::MatrixXd Y;
    
};

#endif