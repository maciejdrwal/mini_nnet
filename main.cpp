#include "autograd.h"
#include "utils.h"
#include "matrix.h"
#include "network.h"

#include "mem_usage.h"

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
        x = MLPBlock(x, W1, W2);
        return softmax(x);
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
        x = MLPBlock(x, W1, W2);
        return softmax(x);
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
    //std::ifstream infile("../data/AllCombined.txt");
    utils::Clock startTimer;

    std::set<char> uchars;
    for (std::string line; std::getline(infile, line); )
    {
        if (line.empty()) continue;
        Xs.push_back(utils::strip(line));
        uchars.insert(line.begin(), line.end());
        if (Xs.size() > 10000) break;
    }
    std::default_random_engine rng{};
    std::shuffle(Xs.begin(), Xs.end(), rng);

    std::cout << "Read " << Xs.size() << " input samples.\n";
    std::cout << "Read data took " << startTimer.get() << " secs." << std::endl;

    const int BOS = uchars.size();
    int vocabSize = uchars.size() + 1;
    std::cout << "Vocab size=" << vocabSize << std::endl;
    std::map<char, int> vocab;
    std::map<int, char> vocabRev;
    int index = 0;
    for (auto ch : uchars) 
    {
        vocab[ch] = index;
        vocabRev[index] = ch;
        ++index;
    }
    
    constexpr int numLayers = 4;
    constexpr int dimEmbed = 128;
    constexpr int blockSize = 64;
    constexpr int numHeads = 8;
    struct Weights
    {
        std::array<Matrix, numLayers> initialize(Tape& tape, int m, int n)
        {
            return std::apply([&](auto... _) { return std::array{((void)_, Matrix(tape, m, n))...}; }, std::array<int, numLayers>{});
        }

        Weights(Tape& tape, int vocabSize) 
        : _tape(tape)
        , WTE(tape, vocabSize, dimEmbed)
        , WPE(tape, blockSize, dimEmbed)
        , LMH(tape, vocabSize, dimEmbed)
        , WQ(initialize(tape, dimEmbed, dimEmbed))
        , WK(initialize(tape, dimEmbed, dimEmbed))
        , WV(initialize(tape, dimEmbed, dimEmbed))
        , WO(initialize(tape, dimEmbed, dimEmbed))
        , M1(initialize(tape, numHeads * dimEmbed, dimEmbed))
        , M2(initialize(tape, dimEmbed, numHeads * dimEmbed))
        {}

        std::vector<int> getIndices() const
        {
            std::vector<int> result;
            for (const auto& w : {WTE, WPE, LMH})
            {
                const auto indices = w.getIndices();
                result.insert(result.end(), indices.begin(), indices.end());
            }
            for (int i = 0; i < numLayers; ++i)
            {
                for (const auto &w : {WQ[i], WK[i], WV[i], WO[i], M1[i], M2[i]})
                {
                    auto indices = w.getIndices();
                    result.insert(result.end(), indices.begin(), indices.end());
                }
            }
            return result;
        }

        Matrix WTE, WPE, LMH;
        std::array<Matrix, numLayers> WQ, WK, WV, WO, M1, M2;
        Tape& _tape;
    } weights(tape, vocabSize);

    std::cout << mem_usage::mem_usage_summary() << std::endl;

    auto gpt = [&](int tokenId, int posId, 
                   std::array<std::vector<Vect>, numLayers>& keys, 
                   std::array<std::vector<Vect>, numLayers>& values)
    {
        Vect tokenEmb = weights.WTE.at(tokenId);
        Vect posEmb = weights.WPE.at(posId);
        Vect x(weights._tape, std::min(tokenEmb.size(), posEmb.size()));
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = tokenEmb[i] + posEmb[i];
        }
        x = rmsnorm(x);

        for (int i = 0; i < numLayers; ++i)
        {
            x = AttentionBlock(x, weights.WQ[i], weights.WK[i], weights.WV[i], weights.WO[i], keys[i], values[i], numHeads);
            x = MLPBlock(x, weights.M1[i], weights.M2[i]);
        }
        Vect logits = linear(x, weights.LMH);
        return logits;
    };

    auto processDoc = [&](Tape& tape, const std::vector<int>& tokens, int n)
    {
        std::array<std::vector<Vect>, numLayers> keys{}, values{}; // TODO: deque? reserve?
        Value sumLoss = tape.newValue(0.0);
        for (int posId = 0; posId < n; ++posId)
        {
            int tokenId = tokens[posId];
            int targetId = tokens[posId + 1];
            Vect logits = gpt(tokenId, posId, keys, values);
            Vect probs = softmax(logits);

            if (probs.at(targetId).get() < 1e-16)
            {
                auto it1 = vocabRev.find(tokenId);
                auto ch1 = it1 != vocabRev.end() ? std::string{it1->second} : "BOS";
                auto it2 = vocabRev.find(targetId);
                auto ch2 = it2 != vocabRev.end() ? std::string{it2->second} : "BOS";
                std::cout << "WARNING: target probability is zero at pos=" << posId << " token=" << ch1 << " target=" << ch2 << std::endl;
                probs.at(targetId).set(1e-16);
            }
            sumLoss -= probs.at(targetId).log();    
        }
        return sumLoss;
    };

    // training
    auto processBatch = [&](Tape& tape, int step) -> Value
    {
        constexpr int batchSize = 10;
        int totalN = 0;
        Value batchLoss = tape.newValue(0.0);
        const int bs = step * batchSize;
        for (int i = bs; i < bs + batchSize; ++i)
        {
            const std::string& doc = Xs[i % Xs.size()];
            std::vector<int> tokens{ BOS };
            for (auto ch : doc) tokens.push_back(vocab.at(ch));
            tokens.push_back(BOS);
            int n = std::min(static_cast<size_t>(blockSize), tokens.size() - 1);
            totalN += n;

            std::cout << "Processing doc " << i << ": " << doc << std::endl;
    
            batchLoss += processDoc(tape, tokens, n);
        }
        Value N = tape.newValue(totalN);
        return batchLoss / N;
    };

    std::cout << "Total number of weights: " << weights.getIndices().size() << std::endl;
    training(processBatch, tape, weights.getIndices(), 200);

    // inference
    //std::random_device rd{};
    std::mt19937 gen{123};
    
    Value temperature = tape.newValue(0.5);
    for (int sampleIdx = 0; sampleIdx < 40; ++sampleIdx)
    {
        std::array<std::vector<Vect>, numLayers> keys, values; // TODO: deque? reserve?
        int tokenId = BOS;
        std::string sample;
        for (int posId = 0; posId < blockSize; ++posId)
        {
            Vect logits = gpt(tokenId, posId, keys, values);
            for (int i = 0; i < logits.size(); ++i)
            {
                logits[i] = logits[i] / temperature;
            }
            Vect probs = softmax(logits);
            std::vector<double> probsValues(probs.size(), 0.0); 
            for (int i = 0; i < probs.size(); ++i)
            {
                probsValues[i] = probs[i].get();
            }
            std::discrete_distribution<int> distribution(probsValues.begin(), probsValues.end());
            tokenId = distribution(gen);
            if (tokenId == BOS)
            {
                break;
            }
            sample += vocabRev.at(tokenId);
        }
        std::cout << "Sample #" << sampleIdx << ": " << sample << std::endl; 
    }

}

// TODO:
// 1. storing/loading weights to/from files
// 2. collecting loss (and possibly other statistics) from each iteration for analysis
// 3. parallel processing

int main()
{
    //testIris();
    //testMnist();
    testGPT();
}
