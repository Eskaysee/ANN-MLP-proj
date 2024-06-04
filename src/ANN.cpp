#include "ANN.hpp"

namespace CLNSIH001{
    using namespace std;
    
    Neuron::Neuron(std::string label, double val): name(label), value(val){}
    Neuron::Neuron(std::string label): name(label){}
    void Neuron::setValue(double val){value = val;}
    void Neuron::addW(Weights weigh){w.push_back(weigh);}

    Weights::Weights(Neuron &s, double w): src(s), weighs(w){}
    Neuron Weights::getSrc(){return src;}

    ANN::ANN(double speed){
        learnRate = speed;
        expected = 0.36;
        init();
    }
    void ANN::init(){
        //Bias
        Neuron b1("Bias", 1), b2("Bias", 1);
        Weights bw1(b1, 0.1), bw2(b1, -0.3), bw3(b2, -0.3);
        hid1.addW(bw1); hid2.addW(bw2); out.addW(bw3);
        //hid1
        Weights w11(in1, 0.1), w21(in2, 0.2), w31(in3, 0.5);
        hid1.addW(w11); hid1.addW(w21); hid1.addW(w31);
        //hid2
        Weights w12(in1, -0.4), w22(in2, 1.0), w32(in3, -0.6);
        hid2.addW(w12); hid2.addW(w22); hid1.addW(w32);
        //out
        Weights w1(hid1, 0.8), w2(hid2, 1.0);
        out.addW(w1); out.addW(w2);
    }
    double ANN::delta(double rate, Neuron p, Neuron source){
        double Wchange = rate * (expected - p.value) * source.value;
        return Wchange;
    }
    double ANN::sigmoidActv(Neuron p){
        double sum = 0.0;
        for (int i=0; i<p.w.size(); i++){
            Weights heavy = p.w.at(i);
            sum += heavy.weighs * heavy.src.value;
        }
        sum = 1/(1+exp(-1*sum));
        cout << "Value: " << sum << endl;
        return sum;
    }
    double ANN::MSE(Neuron p){
        double result = (expected-p.value)*((expected-p.value));
        return result;
    }
    void ANN::view(){
        Neuron continuous[3] = {hid1, hid2, out};
        for (Neuron &n : continuous){
            cout << "--------------------------" << endl;
            cout << "   \t" << n.name << endl;
            cout << '\n' << endl;
            if (n.name == "Output"){
                cout << "Target: " << expected << endl;
                n.w.at(1).src = continuous[0];
                n.w.at(2).src = continuous[1];
            }
            n.setValue(sigmoidActv(n));
            if (n.name == "Output")
                cout << "MSE: " << MSE(n) << endl;
            cout << "--------------------------" << endl;
        }
    }
}

int main(){
    CLNSIH001::ANN ann(0.8);
    ann.view();
    return 0;
}