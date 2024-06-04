compile:
	g++ -o bin/MLP src/MLP.cpp --std=c++11
	g++ -o bin/ANN src/ANN.cpp --std=c++11

Part1:
	./bin/MLP
Part2:
	./bin/ANN

clean:
	rm bin/MLP bin/ANN