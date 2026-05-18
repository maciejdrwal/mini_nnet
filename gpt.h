#pragma once

#include "autograd.h"
#include "matrix.h"
#include "network.h"

#include <array>
#include <map>
#include <set>
#include <vector>
#include <iostream> // TODO remove

namespace mininnet
{
    template <int numLayers, int dimEmbed, int blockSize, int numHeads>
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
        , M1(initialize(tape, 4 * dimEmbed, dimEmbed))
        , M2(initialize(tape, dimEmbed, 4 * dimEmbed))
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
    };

    class GPT : public NeuralNetwork
    {
        static constexpr int numLayers = 2;
        static constexpr int dimEmbed = 64;
        static constexpr int blockSize = 64;
        static constexpr int numHeads = 4;

    public:
        GPT(Tape &tape, const std::vector<std::string> &Xs, const std::set<char> &uchars) 
        : NeuralNetwork(tape), _Xs(Xs), _weights(tape, makeVocab(uchars)), _BOS(uchars.size())
        {
            _paramIndices = _weights.getIndices();
            std::cout << "Total number of weights: " << _paramIndices.size() << std::endl;
        }
        
        Value processBatch(int step) override;
        Vect evaluate(Vect& x) const override;

        using KeysType = std::array<std::vector<Vect>, numLayers>;
        using ValuesType = std::array<std::vector<Vect>, numLayers>;

        Vect gpt(int tokenId, int posId, KeysType& keys, ValuesType& values) const;
        Value processDoc(const std::vector<int>& tokens, int n);
        
        void sample(int numSamples, double t) const;

    protected:
        const std::vector<std::string>& _Xs;
        std::map<char, int> _vocab;
        std::map<int, char> _vocabRev;
        const int _BOS;

        Weights<numLayers, dimEmbed, blockSize, numHeads> _weights;

        mutable KeysType* _internalKeys = nullptr;
        mutable ValuesType* _internalValues = nullptr;

        int makeVocab(const std::set<char>& uchars);
    };

}   // namespace mininnet
