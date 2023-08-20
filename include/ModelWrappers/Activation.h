#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>

#include "Layer.h"

class Activation : public Layer {
    
    public:
        Activation();
        virtual Eigen::MatrixXd getPredictions() const;
        
};

#endif