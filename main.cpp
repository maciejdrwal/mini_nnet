#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <ranges>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;

using namespace mininnet;


void testAutograd()
{
    Tape tape;

    Value x = tape.newValue(0.5);
    Value y = tape.newValue(4.2);
    Value z = x * y + x.sin();

    auto grad = z.backward();

    std::cout << "z = " << z.get() 
              << "\ndz/dx = " << grad.wrt(x)
              << "\ndz/dy = " << grad.wrt(y) << std::endl;

}

void testIris()
{
    Tape tape;

    std::map<std::string, int> labels{{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
    
    std::vector<std::pair<std::vector<double>, int>> Xs;
    std::ifstream infile("../iris/iris.data");

    Clock::time_point startTimer(Clock::now());
    for (std::string line; std::getline(infile, line); )
    {
        const auto entries = utils::split(line, ",");
        std::vector<double> row;
        row.reserve(entries.size() - 1);
        int y = labels.at(entries[4].c_str());
        for (int i = 0; i < 4; ++i)
        {
            row.emplace_back(std::atof(entries[i].c_str()));
        }
        Xs.emplace_back(row, y);
    }

    auto rng = std::default_random_engine {};
    std::shuffle(Xs.begin(), Xs.end(), rng);
    
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - startTimer).count() << " secs." << std::endl;

    Matrix W1(tape, 10, 4), W2(tape, 3, 10);
    std::vector<int> paramIndices = W1.getIndices();
    auto W2Indices = W2.getIndices();
    paramIndices.insert(paramIndices.end(), W2Indices.begin(), W2Indices.end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };

    training(mlp, Xs, tape, paramIndices);

    inference(mlp, Xs,tape);
}

void testMnist()
{
    Tape tape;

    std::vector<std::pair<std::vector<double>, int>> Xs;
    std::ifstream infile("../mnist_train.csv");

    Clock::time_point startTimer(Clock::now());
    for (std::string line; std::getline(infile, line); )
    {
        const auto entries = utils::split(line, ",");
        int y = std::atoi(entries[0].c_str());
        std::vector<double> row;
        row.reserve(entries.size() - 1);
        for (int i = 1; i < entries.size(); ++i)
        {
            row.emplace_back(std::atof(entries[i].c_str()));
        }
        Xs.emplace_back(row, y);
        //TODO
        if (Xs.size() >= 1000) break;
    }
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - startTimer).count() << " secs." << std::endl;

    Matrix W1(tape, 100, 784), W2(tape, 10, 100);
    std::vector<int> paramIndices = W1.getIndices();
    auto W2Indices = W2.getIndices();
    paramIndices.insert(paramIndices.end(), W2Indices.begin(), W2Indices.end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };

    training(mlp, Xs, tape, paramIndices);

    inference(mlp, Xs, tape);
}


int main()
{
    //testAutograd();
    //testMnist();
    testIris();
}
