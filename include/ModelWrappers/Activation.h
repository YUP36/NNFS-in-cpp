#ifndef ACTIVATION_H
#define ACTIVAITON_H

#include <Eigen/Dense>

class Activation {
    
    public:
        Activation();
        virtual Eigen::MatrixXd getPredictions() const;
        
};

#endif