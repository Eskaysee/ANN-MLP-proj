#ifndef ANN_HPP
#define ANN_HPP
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

namespace CLNSIH001{
    class ANN;
    class Weights;
    class Neuron{
        std::string name;
        double value;
        std::vector<Weights> w;
        friend class ANN;
        public:
            Neuron(std::string label, double val);
            Neuron(std::string label);
            void setValue(double val);
            void addW(Weights weigh);
    };
    class Weights{
        Neuron src{"",0.0}; //source
        double weighs=0.0;
        friend class ANN;
        public:
            Weights(Neuron &s, double w);
            Neuron getSrc();
    };
    class ANN{
        Neuron in1{"input1", 1.3}, in2{"input2",2.7}, in3{"input3",0.8};
        Neuron hid1{"Hidden1"}, hid2{"Hidden2"}, out{"Output"};
        double learnRate, expected;
        void init();
        double delta(double rate, Neuron p, Neuron source);
        double sigmoidActv(Neuron p);
        double MSE(Neuron p);
        public:
            ANN(double speed);
            void view();
    };
}

#endif