#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"

#include <iostream>
#include <fstream>
#include <map>
#include <ranges>
#include <algorithm>

using namespace mininnet;

void readDataFromFile(std::ifstream& infile, LabeledData& Xs, 
                      int sampleLimit = std::numeric_limits<int>::max(), 
                      int labelPos = 0)
{
    for (std::string line; std::getline(infile, line); )
    {
        const auto entries = utils::split(line, ",");
        int y = std::atoi(entries[labelPos == -1 ? entries.size() - 1 : labelPos].c_str());
        std::vector<double> row;
        row.reserve(entries.size() - 1);
        for (int i = 1; i < entries.size(); ++i)
        {
            row.emplace_back(std::atof(entries[i].c_str()));
        }
        Xs.emplace_back(row, y);
        if (Xs.size() >= sampleLimit) break;
    }
}

void testIris()
{
    Tape tape;

    std::map<std::string, int> labels{{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};
    
    LabeledData Xs;
    std::ifstream infile("../iris/iris.data");

    utils::Clock startTimer;
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

    std::default_random_engine rng{};
    std::shuffle(Xs.begin(), Xs.end(), rng);
    
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    Matrix W1(tape, 7, 4), W2(tape, 3, 7);
    std::vector<int> paramIndices = W1.getIndices();
    auto W2Indices = W2.getIndices();
    paramIndices.insert(paramIndices.end(), W2Indices.begin(), W2Indices.end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };

    inference(mlp, Xs, tape);

    training(mlp, Xs, tape, paramIndices);

    inference(mlp, Xs, tape);
}

void testMnist()
{
    Tape tape;

    LabeledData Xs;
    std::ifstream infile("../mnist_train.csv");

    utils::Clock startTimer;
    readDataFromFile(infile, Xs, 3000);
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    Matrix W1(tape, 5, 784), W2(tape, 10, 5);
    std::vector<int> paramIndices = W1.getIndices();
    auto W2Indices = W2.getIndices();
    paramIndices.insert(paramIndices.end(), W2Indices.begin(), W2Indices.end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };

    inference(mlp, Xs, tape);

    training(mlp, Xs, tape, paramIndices, 150, 1);

    LabeledData XsTest;
    std::ifstream infileTest("../mnist_test.csv");
    utils::Clock startTimer1;
    readDataFromFile(infileTest, XsTest, 1000);
    std::cout << "Read " << XsTest.size() << " test samples.\n";
    std::cout << "Read data took " << startTimer1.get() << " secs." << std::endl;

    inference(mlp, XsTest, tape);
}


int main()
{
    testMnist();
    //testIris();
}
