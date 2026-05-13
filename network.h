#pragma once

#include "matrix.h"

#include <functional>

namespace mininnet
{
    using LabeledData = std::vector<std::pair<std::vector<double>, int>>;

    Vect linear(Vect& x, Matrix& W);
    Vect rmsnorm(Vect& x);
    Vect softmax(Vect& x);
    Vect MLPBlock(Vect& x, Matrix& W1, Matrix& W2);
    Vect AttentionBlock(Vect& x, Matrix& Wq, Matrix& Wk, Matrix& Wv, Matrix& Wo,
        std::vector<Vect>& keys, std::vector<Vect>& values, 
        int numHeads = 4);

    void training(const std::function<Value(Tape&, int)>& processBatch, 
                  Tape& tape,             
                  const std::vector<int>& paramIndices, 
                  int numSteps = 1000);

    void inference(const std::function<Vect(Vect&)>& network, 
                   const LabeledData& Xs, 
                   Tape& tape);

}   // namespace mininnet