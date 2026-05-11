#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"

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

void readDataFromFile(std::ifstream& infile, LabeledData& Xs, 
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

Value processBatchSupervised(const std::function<Vect(Vect &)> &network,
                             const LabeledData &Xs, Tape &tape, int step, int batchSize)
{
    Value n = tape.newValue(batchSize);
    Value sumLoss = tape.newValue(0.0);
    for (int j = 0; j < batchSize; ++j)
    {
        int sampleId = (step * batchSize + j) % Xs.size();
        const auto &xs = Xs[sampleId];
        int labelId = xs.second;
        Vect x(tape, {xs.first.begin(), xs.first.end()});
        auto probs = network(x);
        sumLoss -= probs[labelId].log();
    }
    Value loss = sumLoss / n;
    return loss;
}

void testIris()
{
    Tape tape;

    auto irisLabelConv = [](const std::string& data) -> double
    {
        static const std::map<std::string, double> labels{{"Iris-setosa", 0.}, {"Iris-versicolor", 1.0}, {"Iris-virginica", 2.0}};
        return labels.at(data.c_str());
    };
    LabeledData Xs;
    std::ifstream infile("../data/iris.data");

    utils::Clock startTimer;
    readDataFromFile(infile, Xs, 150, irisLabelConv, defaultDataConv, 4);

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
    auto processBatch = [&](Tape& tape, int step) -> Value
    {
        return processBatchSupervised(mlp, Xs, tape, step, 150);
    };

    inference(mlp, Xs, tape);
    training(processBatch, tape, paramIndices, 500);
    inference(mlp, Xs, tape);
}

void testMnist()
{
    Tape tape;

    LabeledData Xs;
    std::ifstream infile("../data/mnist_train.csv");

    utils::Clock startTimer;
    readDataFromFile(infile, Xs, 10000);
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    Matrix W1(tape, 400, 784), W2(tape, 10, 400);
    std::vector<int> paramIndices = W1.getIndices();
    auto W2Indices = W2.getIndices();
    paramIndices.insert(paramIndices.end(), W2Indices.begin(), W2Indices.end());

    auto mlp = [&](Vect& x)
    {
        return MLPBlock(x, W1, W2);
    };
    auto processBatch = [&](Tape& tape, int step) -> Value
    {
        return processBatchSupervised(mlp, Xs, tape, step, 200);
    };

    training(processBatch, tape, paramIndices, 10);

    LabeledData XsTest;
    std::ifstream infileTest("../data/mnist_test.csv");
    utils::Clock startTimer1;
    readDataFromFile(infileTest, XsTest, 1000);
    std::cout << "Read " << XsTest.size() << " test samples.\n";
    std::cout << "Read data took " << startTimer1.get() << " secs." << std::endl;

    inference(mlp, XsTest, tape);
}

void testGPT()
{
    Tape tape;

    std::vector<std::string> Xs;
    std::ifstream infile("../data/movie_titles.txt");
    utils::Clock startTimer;

    std::set<char> uchars;
    for (std::string line; std::getline(infile, line); )
    {
        Xs.push_back(utils::strip(line));
        uchars.insert(line.begin(), line.end());
    }
    std::default_random_engine rng{};
    std::shuffle(Xs.begin(), Xs.end(), rng);

    std::cout << "Vocab size=" << uchars.size() << std::endl;
    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    std::map<char, int> vocab;
    int index = 0;
    for (auto ch : uchars) vocab[ch] = index++;
    const int BOS = vocab.size();
    constexpr int numLayers = 1;
    constexpr int dimEmbed = 16;
    constexpr int blockSize = 16;
    constexpr int numHeads = 4;
    struct Weights
    {
        Weights(Tape& tape, int vocabSize) 
        : WTE(tape, vocabSize, dimEmbed),
          WPE(tape, blockSize, dimEmbed),
          LMH(tape, vocabSize, dimEmbed),
          WQ(tape, dimEmbed, dimEmbed),
          WK(tape, dimEmbed, dimEmbed), 
          WV(tape, dimEmbed, dimEmbed),
          WO(tape, dimEmbed, dimEmbed),
          M1(tape, numHeads * dimEmbed, dimEmbed),
          M2(tape, dimEmbed, numHeads * dimEmbed),
          _tape(tape) {}

          std::vector<int> getIndices() const
          {
            std::vector<int> result;
            for (const auto& w : { WTE, WPE, LMH, WQ, WK, WV, WO, M1, M2 })
            {
                auto indices = w.getIndices();
                result.insert(result.end(), indices.begin(), indices.end());
            }
            return result;
          }

        Matrix WTE, WPE, LMH, WQ, WK, WV, WO, M1, M2;
        Tape& _tape;
    } weights(tape, uchars.size());

    auto gpt = [&](int tokenId, int posId, std::vector<Vect>& keys, std::vector<Vect>& values)
    {
        Vect tokenEmb = weights.WTE.at(tokenId);
        Vect posEmb = weights.WPE.at(posId);
        Vect x(weights._tape, std::min(tokenEmb.size(), posEmb.size()));
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = tokenEmb[i] + posEmb[i];
        }
        x = rmsnorm(x);

        // TODO: repeat for numLayers
        x = AttentionBlock(x, weights.WQ, weights.WK, weights.WV, weights.WO, keys, values, numHeads);
        x = MLPBlock(x, weights.M1, weights.M2);
        
        x = linear(x, weights.LMH);
        return x;
    };

    // training
    auto processBatch = [&](Tape& tape, int step) -> Value
    {
        std::string doc = Xs[step % Xs.size()];
        std::vector<int> tokens{ BOS };
        for (auto ch : doc) tokens.push_back(vocab.at(ch));
        int n = std::min(static_cast<size_t>(blockSize), tokens.size() - 1);
        
        std::vector<Vect> keys, values;
        //Vect losses(tape, n);
        Value sumLoss = tape.newValue(0.0);
        for (int posId = 0; posId < n; ++posId)
        {
            int tokenId = tokens[posId];
            int targetId = tokens[posId + 1];
            Vect logits = gpt(tokenId, posId, keys, values);
            Vect probs = softmax(logits);
            sumLoss -= probs.at(targetId).log();
        }
        Value N = tape.newValue(n);
        sumLoss = sumLoss / N;
        return sumLoss;
    };

    training(processBatch, tape, weights.getIndices(), 1);
}

// TODO:
// 1. storing/loading weights to/from files
// 2. collecting loss (and possibly other statistics) from each iteration for analysis
// 3. parallel processing

int main()
{
    //testIris();
    testMnist();
    //testGPT();
}
