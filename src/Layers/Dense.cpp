#include "../../include/Layers/Dense.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

Dense::Dense(int numInputs, int numNeurons, double l1w, double l1b, double l2w, double l2b){
    input = nullptr;
    output = nullptr;

    weights = new MatrixXd(numInputs, numNeurons);
    *weights = 0.1 * MatrixXd::Random(numInputs, numNeurons);

    biases = new RowVectorXd(1, numNeurons);
    *biases = RowVectorXd::Zero(1, numNeurons);

    dinputs = nullptr;
    dweights = new MatrixXd(numInputs, numNeurons);
    dbiases = new RowVectorXd(1, numNeurons);

    weightMomentum = new MatrixXd(numInputs, numNeurons);
    *weightMomentum = MatrixXd::Zero(numInputs, numNeurons);
    biasMomentum = new RowVectorXd(1, numNeurons);
    *biasMomentum = RowVectorXd::Zero(1, numNeurons);

    weightCache = new MatrixXd(numInputs, numNeurons);
    *weightCache = MatrixXd::Zero(numInputs, numNeurons);
    biasCache = new RowVectorXd(1, numNeurons);
    *biasCache = RowVectorXd::Zero(1, numNeurons);

    lambdaL1Weight = l1w;
    lambdaL1Bias = l1b;
    lambdaL2Weight = l2w;
    lambdaL2Bias = l2b;
}

std::string Dense::getName() const {
    return "Dense";
}

ostream& operator<<(ostream& os, const Dense& layer) {
    os << "Weights:\n" << layer.getWeights() << std::endl << "Biases:\n" << layer.getBiases() << endl;
    return os;
}

MatrixXd* Dense::getWeights() const {
    return weights;
}

void Dense::setWeights(MatrixXd* newWeights) {
    *weights = *newWeights;
}

void Dense::updateWeights(MatrixXd* weightsUpdate) {
    *weights += *weightsUpdate;
}

RowVectorXd* Dense::getBiases() const {
    return biases;
}

void Dense::setBiases(RowVectorXd* newBiases) {
    *biases = *newBiases;
}

void Dense::updateBiases(RowVectorXd* biasesUpdate) {
    *biases += *biasesUpdate;
}


void Dense::forward(MatrixXd* in) {
    input = in;
    if(!output || (input->rows() != output->rows())) output = new MatrixXd(input->rows(), weights->cols());
    *output = ((*input) * (*weights)).rowwise() + (*biases);
}

MatrixXd* Dense::getOutput() const {
    return output;
}

void Dense::backward(MatrixXd* dvalues) {
    *dweights = input->transpose() * (*dvalues);
    *dbiases = dvalues->colwise().sum();
    
    if(lambdaL1Weight > 0) {
        *dweights += lambdaL1Weight * weights->unaryExpr([](double x){return (x > 0) ? 1.0 : -1.0;});
    }
    if(lambdaL1Bias > 0) {
        *dbiases += lambdaL1Bias * biases->unaryExpr([](double x){return (x > 0) ? 1.0 : -1.0;});
    }
    if(lambdaL2Weight > 0) {
        *dweights += 2 * lambdaL2Weight * *weights;
    }
    if(lambdaL2Bias > 0) {
        *dbiases += 2 * lambdaL2Bias * *biases;
    }

    if(!dinputs || (dvalues->rows() != dinputs->rows())) dinputs = new MatrixXd(dvalues->rows(), weights->rows());
    *dinputs = (*dvalues) * weights->transpose();
}

MatrixXd* Dense::getDinputs() const {
    return dinputs;
}

MatrixXd* Dense::getDweights() const {
    return dweights;
}

RowVectorXd* Dense::getDbiases() const {
    return dbiases;
}

MatrixXd* Dense::getWeightMomentum() const {
    return weightMomentum;
}

void Dense::setWeightMomentum(MatrixXd* newWeightMomentum) {
    *weightMomentum = *newWeightMomentum;
}

RowVectorXd* Dense::getBiasMomentum() const {
    return biasMomentum;
}

void Dense::setBiasMomentum(RowVectorXd* newBiasMomentum) {
    *biasMomentum = *newBiasMomentum;
}

MatrixXd* Dense::getWeightCache() const {
    return weightCache;
}

void Dense::setWeightCache(MatrixXd* update) {
    *weightCache = *update;
}

void Dense::updateWeightCache(MatrixXd* update) {
    *weightCache += *update;
}

RowVectorXd* Dense::getBiasCache() const {
    return biasCache;
}

void Dense::setBiasCache(RowVectorXd* update) {
    *biasCache = *update;
}

void Dense::updateBiasCache(RowVectorXd* update) {
    *biasCache += *update;
}

double Dense::getLambdaL1Weight() const {
    return lambdaL1Weight;
}

double Dense::getLambdaL1Bias() const {
    return lambdaL1Bias;
}

double Dense::getLambdaL2Weight() const {
    return lambdaL2Weight;
}

double Dense::getLambdaL2Bias() const {
    return lambdaL2Bias;
}