#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"

#include <iostream>
#include <fstream>
#include <chrono>

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

void testMnist()
{
    Tape tape;

    std::vector<double> ys;
    std::vector<std::vector<double>> Xs;
    std::ifstream infile("../mnist_train.csv");

    Clock::time_point startTimer(Clock::now());
    for (std::string line; std::getline(infile, line); )
    {
        const auto entries = utils::split(line, ",");
        ys.emplace_back(std::atof(entries[0].c_str()));
        std::vector<double> row;
        row.reserve(entries.size() - 1);
        for (int i = 1; i < entries.size(); ++i)
        {
            row.emplace_back(std::atof(entries[i].c_str()));
        }
        Xs.emplace_back(row);
        //TODO
        if (Xs.size() >= 500) break;
    }
    std::cout << "Read " << Xs.size() << " input samples with " << ys.size() << " labels.\n";
    std::cout << "Read data took " << std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - startTimer).count() << " secs." << std::endl;

    Matrix W1(tape, 20, 784), W2(tape, 10, 20);
    std::vector<int> paramIndices = W1.getIndices();
    paramIndices.insert(paramIndices.end(), W2.getIndices().begin(), W2.getIndices().end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };

    training(mlp, Xs, ys, tape, paramIndices);
}


int main()
{
    //testAutograd();
    testMnist();
}
