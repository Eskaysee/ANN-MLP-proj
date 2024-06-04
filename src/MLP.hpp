#ifndef MLP_HPP
#define MLP_HPP
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

namespace CLNSIH001{
    class Weights;      //Weights class must be know to Neuron before its decleration
    class Perceptron;   //Same goes for Perceptron when it comes to Neuron and Weights
    class Neuron{
        std::string name;
        double value;
        std::vector<Weights> w;
        double target;
        friend class Perceptron;
        public:
            Neuron(std::string label);
            void updateTarget();
    };
    class Weights{
        Neuron src{""}; //source
        double weighs=0.0;
        void setWeights(Neuron &s, double w);
        friend class Perceptron;
        public:
            Neuron getSrc();
    };
    class Perceptron{
        double learnRate;
        //double trainData[8][3] = {{0.0,0.0,0.0},{0.8,0.2,1},{0.6,0.6,0},{0.0,1.0,1.0},{0.3,0.7,1.0},{1.0,0.0,1.0},{0.4,0.4,0},{1.0,1.0,0.0}};
        //double trainData[4][3] = {{0.0,0.0,0.0},{0.0,1.0,1.0},{1.0,0.0,1.0},{1.0,1.0,0.0}};
        double trainData[12][3] = {{0.0,0.0,0.0},{0.8,0.2,1.0},{0.6,0.6,0.0},{0.0,1.0,1.0},{0.3,0.7,1.0},{1.0,0.0,1.0},{0.4,0.4,0.0},{1.0,1.0,0.0},{1.0,1.0,0.0},{1.0,0.0,1.0},{0.0,0.0,0.0},{0.0,1.0,1.0}};
        Neuron in1{"input1"}, in2{"input2"}, p1{"OR"}, p2{"NAND"}, out{"AND"};
        void init();
        void train();
        double output(Neuron perceptron);
        double delta(double rate, Neuron p, Neuron source);
        void update(Neuron arr[]);
        public:
            Perceptron(double learnSpeed);
            double test(double input1, double input2);
    };
    //QUESTIONS: Do all the perceptrons use the same rule?
}
#endif