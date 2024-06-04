#include "MLP.hpp"

namespace CLNSIH001{
    using namespace std;
    
    Neuron::Neuron(string label):name(label), value(0.0){}
    void Neuron::updateTarget(){
        double input1 = w.at(1).getSrc().value;
        double input2 = w.at(2).getSrc().value;
        if (name == "OR"){
            if (input1 < 0.5 && input2 < 0.5)
                target = 0.0;
            else
                target = 1.0;
        }
        else if (name == "NAND"){
            if (input1 > 0.5 && input2 > 0.5)
                target = 0.0;
            else
                target = 1.0;
        }
        else if (name == "AND"){
            if (input1 >= 0.6 && input2 >= 0.6)
                target = 1.0;
            else
                target = 0.0;
        }
    }

    Neuron Weights::getSrc(){
        return src;
    }
    void Weights::setWeights(Neuron &s, double w){
        src = s;
        weighs = w;
    }

    Perceptron::Perceptron(double learnSpeed){
        init();
        learnRate = learnSpeed;
        train();
    }
    void Perceptron::init(){
        srand(time(NULL));
        Weights w11, w12, w21, w22, w1, w2;
        //bias terms
        Neuron b1{"Bias"}, b2{"Bias"};
        Weights bw1, bw2;
        b1.value = 1;
        bw1.setWeights(b1,0.6);
        p1.w.push_back(bw1);
        p2.w.push_back(bw1);
        b2.value = 1;
        bw2.setWeights(b2,-5);
        out.w.push_back(bw2);
        //perceptron Neuron incoming weights
        w11.setWeights(in1, (double)((rand()%11)/10.0));
        p1.w.push_back(w11);
        w12.setWeights(in1, (double)((rand()%11)/10.0));
        p2.w.push_back(w12);
        w21.setWeights(in2, (double)((rand()%11)/10.0));
        p1.w.push_back(w21);
        w22.setWeights(in2, (double)((rand()%11)/10.0));
        p2.w.push_back(w21);
        //output Neuron incoming weights
        w1.setWeights(p1, (double)((rand()%11)/10.0));
        out.w.push_back(w1);
        w2.setWeights(p2, (double)((rand()%11)/10.0));
        out.w.push_back(w2);
    }
    void Perceptron::train(){
        int iterations = 0;
        bool err = true; //error
        Neuron Perceptrons[3] = {p1, p2, out};
        while (err && iterations<12){
            err = false;
            for (int i=0; i<12; ++i){
                //input neurons
                in1.value = trainData[i][0];
                in2.value = trainData[i][1];
                Perceptrons[2].target = trainData[i][2];
                //cout << "-----------------------------------------------------------" << endl;
                //cout << "inputs: " << in1.value << " & " << in2.value << endl;
                for (Neuron &p : Perceptrons){
                    //cout << p.name << endl;
                    if (p.name != "AND"){
                        p.w.at(1).src = in1;
                        p.w.at(2).src = in2;
                    }
                    else{
                        p.w.at(1).src = Perceptrons[0];
                        p.w.at(2).src = Perceptrons[1];
                    }
                    p.updateTarget();
                    //cout << "Expectation: " << p.target << endl;
                    p.value = output(p);
                    //cout << "Reality: " << p.value << endl;
                    if (p.target != p.value){
                        err = true;
                        //cout << "WEIGHT UPDATE!!!!!!!!!!!" << endl;
                        for (Weights &link : p.w){
                            if (link.src.name == "Bias")
                                link.weighs += learnRate*(p.target-p.value);
                            else
                                link.weighs += delta(learnRate, p, link.src);
                        }
                    }
                    //cout << '\n' << endl;
                }
            }
            ++iterations;
        }
        update(Perceptrons);
    }
    double Perceptron::output(Neuron p){
        double sum = 0.0;
        for (int i=0; i<p.w.size(); i++){
            Weights heavy = p.w.at(i);
            sum += heavy.weighs * heavy.src.value;
        }
        //Rule
        //cout << "sum is " << sum << endl;
        if (sum <= 0.0)
            sum = 0.0;
        else
            sum = 1.0;
        return sum;
    }
    double Perceptron::delta(double rate, Neuron p, Neuron source){
        double Wchange = rate * (p.target - p.value) * source.value;
        return Wchange;
    }
    void Perceptron::update(Neuron arr[]){
        p1 = arr[0];
        p2 = arr[1];
        out = arr[2];
        ////////
        /*Neuron Perceptrons[3] = {p1, p2, out};
        for (Neuron &p : Perceptrons){
            cout << "-----------------------" << endl;
            cout << p.name << endl;
            cout <<p.w.at(1).src.name << " w: " << p.w.at(1).weighs << " & v: " << p.w.at(1).src.value << endl;
            cout <<p.w.at(2).src.name << " w: " << p.w.at(2).weighs << " & v: " << p.w.at(2).src.value << endl;
            cout <<p.w.at(0).src.name << " w: " << p.w.at(0).weighs << " & v: " << p.w.at(0).src.value << endl;
            cout << '\n' << endl;
            cout << "-----------------------" << endl;
        }*/
    }
    double Perceptron::test(double input1, double input2){
        in1.value = input1;
        in2.value = input2;
        //cout << "-------------------------------------" << endl;        
        //cout << in1.name << ": " << input1 << " & " << in2.name << ": " << input2 << endl;
        Neuron Perceptrons[3] = {p1, p2, out};
        for (Neuron &p : Perceptrons){
            if (p.name != "AND"){
                p.w.at(1).src = in1;
                p.w.at(2).src = in2;
            }
            else{
                p.w.at(1).src = Perceptrons[0];
                p.w.at(2).src = Perceptrons[1];
            }
            p.value = output(p);
            //cout << p.name << ": " << p.value << endl;
        }
        update(Perceptrons);
        return out.value;
    }
}

int main(){
    /*CLNSIH001::Perceptron MLP(0.8);
    std::cout << MLP.test(0.0,0.0) << std::endl;
    std::cout << MLP.test(0.0,1.0) << std::endl;
    std::cout << MLP.test(1.0,0.0) << std::endl;
    std::cout << MLP.test(1.0,1.0) << std::endl;*/
    std::cout << "Please Enter Learning Rate:" << std::endl; 
    double learningSpeed = 0.8;
    std::cin >> learningSpeed;
    CLNSIH001::Perceptron MLP(learningSpeed);
    std::cout << "-------------------------------------" << std::endl;        
    std::cout << "input 1: 0.5 & input 2: 0.4" << std::endl;
    std::cout << "result: " << MLP.test(0.5,0.4) << std::endl;

    std::cout << "-------------------------------------" << std::endl;        
    std::cout << "input 1: 0.9 & input 2: 0.9" << std::endl;
    std::cout << "result: " << MLP.test(0.9,0.9) << std::endl;

    std::cout << "-------------------------------------" << std::endl;        
    std::cout << "input 1: 0.5 & input 2: 0.6" << std::endl;
    std::cout << "result: " << MLP.test(0.5,0.6) << std::endl;

    std::cout << "-------------------------------------" << std::endl;        
    std::cout << "input 1: 0.1 & input 2: 0.1" << std::endl;
    std::cout << "result: " << MLP.test(0.1,0.1) << std::endl;
}
