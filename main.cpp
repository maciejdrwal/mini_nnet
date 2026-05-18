#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"
#include "gpt.h"

#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <ranges>
#include <algorithm>

using namespace mininnet;

double defaultDataConv(const std::string& data)
{
    return std::atof(data.c_str());
}

void readDataFromFile(std::ifstream& infile, Classifier::LabeledData& Xs,
                      int sampleLimit = std::numeric_limits<int>::max(),
                      const std::function<double(const std::string&)>& labelConverter = defaultDataConv,
                      const std::function<double(const std::string&)>& dataConverter = defaultDataConv,
                      int labelPos = 0)
{
    for (std::string line; std::getline(infile, line); )
    {
        const auto entries = utils::split(line, ",");

        int y = labelConverter(entries[labelPos == -1 ? entries.size() - 1 : labelPos]);
        std::vector<double> row;
        row.reserve(entries.size() - 1);
        for (int i = 0; i < entries.size(); ++i)
        {
            if (i == labelPos) continue;
            row.emplace_back(dataConverter(entries[i]));
        }
        Xs.emplace_back(row, y);
        if (Xs.size() >= sampleLimit) break;
    }
}

void testIris()
{ 
    auto irisLabelConv = [](const std::string& data) -> double
    {
        static const std::map<std::string, double> labels{{"Iris-setosa", 0.}, {"Iris-versicolor", 1.0}, {"Iris-virginica", 2.0}};
        return labels.at(data.c_str());
    };
    Classifier::LabeledData Xs;
    std::ifstream infile("../data/iris.data");

    utils::Clock startTimer;
    readDataFromFile(infile, Xs, 150, irisLabelConv, defaultDataConv, 4);

    std::default_random_engine rng{};
    std::shuffle(Xs.begin(), Xs.end(), rng);

    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    Tape tape;

    struct IrisClassifier : public Classifier
    {
        IrisClassifier(Tape& tape, const LabeledData& Xs) : Classifier(tape, Xs), W1(tape, 7, 4), W2(tape, 3, 7) 
        {
            _paramIndices = W1.getIndices();
            auto W2Indices = W2.getIndices();
            _paramIndices.insert(_paramIndices.end(), W2Indices.begin(), W2Indices.end());
        }
       
        Vect evaluate(Vect& x) const override
        {
            x = MLPBlock(x, W1, W2);
            return softmax(x);
        }
      
        Value processBatch(int step) override
        {
            return processBatchSupervised(step, 150);
        };

    private:
        Matrix W1, W2;
    } iris(tape, Xs);;

    iris.inference(Xs);
    iris.training(500);
    iris.inference(Xs);
}

void testMnist()
{
    Classifier::LabeledData Xs;
    std::ifstream infile("../data/mnist_train.csv");

    utils::Clock startTimer;
    readDataFromFile(infile, Xs, 10000);
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    Tape tape;

    struct MnistClassifier : public Classifier
    {
        MnistClassifier(Tape& tape, const LabeledData& Xs) 
        : Classifier(tape, Xs), W1(tape, 400, 784), W2(tape, 10, 400)
        {
            _paramIndices = W1.getIndices();
            auto W2Indices = W2.getIndices();
            _paramIndices.insert(_paramIndices.end(), W2Indices.begin(), W2Indices.end());
        }

        Vect evaluate(Vect& x) const override
        {
            x = MLPBlock(x, W1, W2);
            return softmax(x);
        }

        Value processBatch(int step) override
        {
            return processBatchSupervised(step, 100);
        };

    private:
        Matrix W1, W2;
    } mnist(tape, Xs);

    mnist.training(10);

    Classifier::LabeledData XsTest;
    std::ifstream infileTest("../data/mnist_test.csv");
    utils::Clock startTimer1;
    readDataFromFile(infileTest, XsTest, 1000);
    std::cout << "Read " << XsTest.size() << " test samples.\n";
    std::cout << "Read data took " << startTimer1.get() << " secs." << std::endl;

    mnist.inference(XsTest);
}

void testGPT()
{
    Tape tape;

    std::vector<std::string> Xs;
    //std::ifstream infile("../data/movie_titles.txt");
    std::ifstream infile("../data/AllCombined_one.txt");
    utils::Clock startTimer;

    std::set<char> uchars;
    for (std::string line; std::getline(infile, line); )
    {
        if (line.empty()) continue;
        // for (auto ch : ".,!@#$%^&*()\"-+_+£~`<>?/:;'\\|–")
        // {
        //     utils::replaceAll(line, std::string{ch}, " ");
        // }
        // utils::replaceAll(line, "   ", " ");
        // utils::replaceAll(line, "  ", " ");
        // line.erase(std::remove_if(line.begin(), line.end(), [](char c){return !isalnum(c) && !isspace(c);}), line.end());
        Xs.push_back(utils::strip(line));
        uchars.insert(line.begin(), line.end());
        line.clear();
    }
    std::default_random_engine rng{};
    std::shuffle(Xs.begin(), Xs.end(), rng);

    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    GPT gpt(tape, Xs, uchars);

    gpt.training(100);
    
    gpt.sample(20, 0.5);
    gpt.sample(20, 0.3);
}

// TODO:
// 1. storing/loading weights to/from files
// 2. collecting loss (and possibly other statistics) from each iteration for analysis
// 3. parallel processing
// 4. proper logging

int main()
{
    //testIris();
    //testMnist();
    testGPT();
}
