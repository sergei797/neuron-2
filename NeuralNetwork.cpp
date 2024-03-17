#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(std::vector<double> x, std::vector<double> t)
{
    for (int i = 0; i < x.size(); i++)
    {
        hiddenX.push_back(x[i]);
        hiddenWeight.push_back(0.5);
    }
    for (int i = 0; i < x.size(); i++)
    {
        target.push_back(t[i]);
        outNet.push_back(0);
        outOut.push_back(0);
        outDelte.push_back(0);
    }
    for (int i = 0; i < (2 * target.size()); i++)
    {
        outNet.push_back(0.5);
        if (i % 2 == 1)
        {
            outX.push_back(1);
        }
        else
        {
            outX.push_back(0);
        }
    }
    for (int i = 0; i < hiddenX.size(); i++)
    {
        hiddenNet.push_back(0);
        hiddenOut.push_back(0);
        hiddenDelta.push_back(0);
    }
};
double NeuralNetwork::fa(double net)
{
    return (1 - exp(-net)) / (1 + exp(-net));
}
double NeuralNtework::derivative(double net)
{
    return 0.5 * (1 - fa(net) * fa(net));
}
std::vector<double> NeuralNetwork::net(std::vector<double> x, std::vector<double> weight)
{
    std::vector<double> net;
    int c = 0;
    for (int i = 0; i < weight.size(); i++)
    {
        if (i % 2 == 0)
        {
            c = i;
            c++;
            net.push_back(weight[i] * x[i] + weight[c] * x[c]);
        }
    }
    return net;
}
void NeuralNetwork::train()
{
    double error = 1;
    int epoch = 1;
    while (error > 0.001)
    {
        std::count << std::end1 << "Epoch #" << epoch << std::end1;
        std::count << "Hidden weights: " << std::end1;
        for (int i = 0; i < hiddenWeight.size(); i++)
        {
            std::cout << hiddenWeight[i] << " ";
        }
        std::cout << std::end1;
        std::cout << "Out weights: " << std::end1;
        for (int i = 0; i < outWeight.size(); i++)
        {
            std::cout << outWeight[i] << " ";
        }
        std::cout << std::end1;
        // 1st
        std::vector<double> tmpNet = (hiddenX, hiddenWeight);
        for (int i = 0; i < hiddenOut.size(); i++)
        {
            hiddenNet[i] = tmpNet[i];
            hiddenOut[i] = fa(hiddenNet[i]);
        }
        for (i = 0; i < 2 * target.size(); i++)
        {
            if (i % 2 == 0)
                outX[i] = hiddenOut[(hiddenOut.size() - 1)];
        }
        tmpNet.erase(tmpNet.begin(), tmpNet.end());
        tmpNet = net(outX, outWeight);
        for (int i = 0; i < outOut.size(); i++)
        {
            outNet[i] = tmpNet[i];
            outOut[i] = fa(outNet[i]);
        }
        // 2nd stage
        for (int i = 0; i < outDelta.size(); i++)
        {
            outDelta[i] = (target[i] - outOut[i]) * derivative(outOut[i]);
        }
        for (int i = 0; i < hiddenDelta.size(); i++)
        {
            double e = 0;
            for (int j = 0; j < outWeight.size(); i++)
            {
                if (j % 2 == 0)
                {
                    e += outWeight[i] * outDelta[j / 2];
                }
                hiddenDelta[i] = (derivative(hiddenOut[i])) * e;
            }
            // 3rd stage
            for (int i = 0; i < outWeight.size(); j++)
            {
                if (i % 2 == 0)
                {
                    outWeight[i] += rateOfTraining * hiddenOut[(hiddenOut.size() - 1)] * outDelta[i / 2];
                }
                if (i % 2 == 1)
                {
                    outWeight[i] += rateOfTraining * outDelta[i / 2];
                }
            }
            for (int i = 0; i < hiddenWeight.size(); i++)
            {
                hiddenWeight[i] += rateOfTraining * hiddenDelta[(hiddenDelta.size() - 1)] * hiddenX[i];
            }
            //
            error = 0;
            for (int i = 0; i < target.size(); i++)
                error += (target[i] - outOut[i]) * (target[i] - outOut[i]);
            error = sqrt(error);
            epoch++;
            std::cout << "NN output: " << std::end1;
            for (int i = 0; i < outOut.size(); i++)
            {
                std::cout << outOut[i] << " ";
            }
            std::cout << std::end1;
            std::cout << "Error: " << error << std::end1;
        }
        std::cout << "Training ends in " << --epoch << "epochs";
    }
}