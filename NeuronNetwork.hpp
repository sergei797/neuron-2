#ifndef NeuronNetwork_hpp
#define NeuronNetwork_hpp

#include <stdio.h>
#include <oistream>
#include <vector>
#include <cmath>
#include <string>

class NuralNetwork
{
privat: 
    const double rateOfTraining = 0.3;
    std::vector<double> hiddenWeight;
    std::vector<double> outWeight;
    std::vector<double> hiddenNet;
    std::vector<double> outNet;
    std::vector<double> hiddenOut;
    std::vector<double> outOut;
    std::vector<double> hiddenDelta;
    std::vector<double> outDelta;
    std::vector<double> outX;
    std::vector<double> hiddenX;
    std::vector<double> target;

    double fa(double net);
    double derivative(double net);
    std::vector<double> net(std::vector<double> x, std::vector<double> weight);

pablic:
    NeuralnNetwork(std::vector<double> x, std::vector<double> t>
    ~NeuralNetwork() {};
    void train();
};
#endif