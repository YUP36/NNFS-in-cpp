#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../Layers/Dense.h"

class Optimizer {

    public:
        Optimizer();
        virtual double getLearningRate();
        virtual void decay();
        virtual void updateParameters(Dense* layer);
        virtual void incrementIteration();

};

#endif