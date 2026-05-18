#include "gpt.h"
#include "network.h"
#include "utils.h"

namespace
{
    std::vector<std::string> splitIntoSentences(const std::string& doc, int maxLength)
    {
        std::vector<std::string> result;
        const auto words = mininnet::utils::split(doc, " ");
        int wordId = 0;
        
        while (true)
        {
            std::string sentence{};
            while (true)
            {
                if (wordId >= words.size() || (sentence.size() + words[wordId].size() >= maxLength - 1))
                {
                    break;
                }
                sentence += words[wordId++] + ' ';
            }
            if (sentence.empty()) break;
            sentence = mininnet::utils::strip(sentence);
            result.emplace_back(std::move(sentence));
        }

        return result;
    }
}

namespace mininnet
{
    int GPT::makeVocab(const std::set<char> &uchars)
    {
        int vocabSize = uchars.size() + 1;
        std::cout << "Vocab size=" << vocabSize << std::endl;
        int index = 0;
        for (auto ch : uchars)
        {
            _vocab[ch] = index;
            _vocabRev[index] = ch;
            std::cout << "char:" << ch << "->" << index << std::endl;
            ++index;
        }
        return vocabSize;
    }

    Vect GPT::evaluate(Vect& x) const
    {
        x = rmsnorm(x);
        for (int i = 0; i < numLayers; ++i)
        {
            x = AttentionBlock(x, _weights.WQ[i], _weights.WK[i], _weights.WV[i], _weights.WO[i], (*_internalKeys)[i], (*_internalValues)[i], numHeads);
            x = MLPBlock(x, _weights.M1[i], _weights.M2[i]);
        }
        Vect logits = linear(x, _weights.LMH);
        return logits;
    }


    Vect GPT::gpt(int tokenId, int posId, KeysType& keys, ValuesType& values) const
    {
        Vect tokenEmb = _weights.WTE.at(tokenId);
        Vect posEmb = _weights.WPE.at(posId);
        Vect x(_tape, std::min(tokenEmb.size(), posEmb.size()));
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = tokenEmb[i] + posEmb[i];
        }
        _internalKeys = &keys;
        _internalValues = &values;
        return evaluate(x);
    }

    Value GPT::processDoc(const std::vector<int>& tokens, int n)
    {
        std::array<std::vector<Vect>, numLayers> keys{}, values{}; // TODO: deque? reserve?
        Value sumLoss = _tape.newValue(0.0);
        for (int posId = 0; posId < n; ++posId)
        {
            int tokenId = tokens[posId];
            int targetId = tokens[posId + 1];
            Vect logits = gpt(tokenId, posId, keys, values);
            Vect probs = softmax(logits);

            if (probs.at(targetId).get() < 1e-16)
            {
                auto it1 = _vocabRev.find(tokenId);
                auto ch1 = it1 != _vocabRev.end() ? std::string{it1->second} : "BOS";
                auto it2 = _vocabRev.find(targetId);
                auto ch2 = it2 != _vocabRev.end() ? std::string{it2->second} : "BOS";
                std::cout << "WARNING: target probability is zero at pos=" << posId << " token=" << ch1 << " target=" << ch2 << std::endl;
                probs.at(targetId).set(1e-16);
            }
            sumLoss -= probs.at(targetId).log();
        }
        return sumLoss;
    }

    Value GPT::processBatch(int step)
    {
        constexpr int batchSize = 1;
        int totalN = 0;
        Value batchLoss = _tape.newValue(0.0);
        const int bs = step * batchSize;
        for (int i = bs; i < bs + batchSize; ++i)
        {
            const std::string& doc = _Xs[i % _Xs.size()];
            //std::cout << "Doc: " << doc << std::endl;

            std::vector<std::string> sentences = splitIntoSentences(doc, blockSize);
            for (const auto& sentence : sentences)
            {
                std::vector<int> tokens{ _BOS };
                for (auto ch : sentence)
                {
                    const auto it = _vocab.find(ch);
                    if (it != _vocab.end())
                    {
                        tokens.push_back(it->second);
                    }
                    else
                    {
                        std::cout << "Warning: token not found: " << ch << " in sentence:" << sentence << std::endl;
                    }
                }
                tokens.push_back(_BOS);

                int n = std::min(static_cast<size_t>(blockSize), tokens.size() - 1);
                totalN += n;

                std::cout << "Processing doc #" << i << " fragment=|" << sentence << "| len=" << sentence.size() << std::endl;

                batchLoss += processDoc(tokens, n);
            }
        }
        Value N = _tape.newValue(totalN);
        return batchLoss / N;
    }
   
    void GPT::sample(int numSamples, double t) const
    {
        //std::random_device rd{};
        std::mt19937 gen{123};
        Value temperature = _tape.newValue(t);
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            std::array<std::vector<Vect>, numLayers> keys, values; // TODO: deque? reserve?
            int tokenId = _BOS;
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
                if (tokenId == _BOS)
                {
                    break;
                }
                sample += _vocabRev.at(tokenId);
            }
            std::cout << "Sample #" << sampleIdx << ": " << sample << std::endl;
        }
    }

}   // namespace mininnet