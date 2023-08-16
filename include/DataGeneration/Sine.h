#ifndef SINE_H
#define SINE_H

#include <iostream>
#include <Eigen/Dense>

class Sine {

    public:
        Sine(int numSamples = 1000);
        friend std::ostream& operator<<(std::ostream& os, const Sine& dataset);
        Eigen::VectorXd getX() const;
        Eigen::VectorXd getY() const;
        
    private:
        Eigen::VectorXd X;
        Eigen::VectorXd Y;

};

#endif