#pragma once

#include "matrix.h"

namespace mininnet
{
    class NeuralNetwork
    {
    public:
        NeuralNetwork(Tape& tape) : _tape(tape) {}
        
        Vect linear(const Vect& x, const Matrix& W) const;
        Vect rmsnorm(const Vect& x) const;
        Vect softmax(const Vect& x) const;
        Vect MLPBlock(Vect& x, const Matrix& W1, const Matrix& W2) const;
        Vect AttentionBlock(Vect& x, const Matrix& Wq, const Matrix& Wk, const Matrix& Wv, const Matrix& Wo,
                            std::vector<Vect> &keys, std::vector<Vect> &values,
                            int numHeads = 4) const;


        virtual Vect evaluate(Vect& x) const = 0;
        virtual Value processBatch(int step) = 0;

        void training(int numSteps = 1000);

    protected:
        Tape& _tape;
        std::vector<int> _paramIndices;
    };

    struct Classifier : public NeuralNetwork
    {
        using LabeledData = std::vector<std::pair<std::vector<double>, int>>;

        Classifier(Tape& tape, const LabeledData& Xs) : NeuralNetwork(tape), _Xs(Xs) {}

        void inference(const LabeledData& Xs) const;

    protected:
        const LabeledData& _Xs;

        Value processBatchSupervised(int step, int batchSize) const;
    };

}   // namespace mininnet