#pragma once

#include "matrix.h"

#include <algorithm>
#include <functional>

namespace mininnet
{
    Vect linear(Vect& x, Matrix& W)
    {
        if (x.size() != W.numCols())
        {
            throw std::runtime_error("linear: invalid dimensions: dim(x)=" + std::to_string(x.size()) +
                                     " cols="  + std::to_string(W.numCols()) + " rows=" + std::to_string(W.numRows()));
        }

        Vect result(x[0].getTape(), W.numRows());
        for (int i = 0; i < W.numRows(); ++i)
        {
            const Vect& wi = W.at(i);
            for (int j = 0; j < wi.size(); ++j)
            {
                result[i] += wi[j] * x[j];
            }
        }
        return result;
    }

    Vect rmsnorm(Vect& x)
    {
        Value ms = x[0].getTape().newValue(0);
        Value eps = x[0].getTape().newValue(1e-5);
        Value d = x[0].getTape().newValue(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            ms += x[i] * x[i];
        }
        ms = ((ms + eps)/ d).pow(0.5);

        Vect result(x[0].getTape(), x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = x[i] * ms; 
        }
        return result;
    }

    Vect softmax(Vect& x)
    {
        double maxElem = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < x.size(); ++i)
        {
            maxElem = std::max(x[i].get(), maxElem);
        }
        Value maxVal = x[0].getTape().newValue(maxElem);
        Value total = x[0].getTape().newValue(0);
        Vect exps(x[0].getTape(), x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            exps[i] = (x[i] - maxVal).exp();
            total += exps[i];
        }
        Vect result(x[0].getTape(), x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = exps[i] / total;
        }
        return result;
    }

    Vect MLPBlock(Vect& x, Matrix& W1, Matrix& W2)
    {
        Vect xResidual = x;
        x = rmsnorm(x);
        x = linear(x, W1);
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = x[i].relu(); 
        }
        x = linear(x, W2);
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] += xResidual[i];
        }
        return softmax(x);
    }

    void training(const std::function<Vect(Vect&)>& network, 
                  const std::vector<std::pair<std::vector<double>, int>>& Xs, 
                  Tape& tape,             
                  const std::vector<int>& paramIndices, 
                  int batchSize = 150,
                  int numSteps = 1000)
    {
        utils::Clock startTimer;
        int step = 0;
        double learningRate = 0.01;
        double beta1 = 0.85;
        double beta2 = 0.99;
        double eps_adam = 1e-8;
        std::vector<double> m(paramIndices.size(), 0.0);  // first moment buffer
        std::vector<double> v(paramIndices.size(), 0.0);  // second moment buffer

        Value n = tape.newValue(batchSize);
        while (step < numSteps)
        {
            std::cout << "Training step: " << step << std::endl;
            Value sumLoss = tape.newValue(0.0);
            for (int j = 0; j < batchSize; ++j)
            {
                int sampleId =  (step * batchSize + j) % Xs.size();
                auto xs = Xs[sampleId];
                int labelId = xs.second;
                Vect x(tape, {xs.first.begin(), xs.first.end()});
                auto probs = network(x);
                sumLoss -= probs[labelId].log();
            }

            Value loss = sumLoss / n;

            std::cout << "Loss=" << loss.get() << std::endl;

            Grad grad = loss.backward();

            int paramCorrCount = 0;
            double paramCorrTotal = 0;

            double lr_t = learningRate * (1 - step / numSteps); // linear learning rate decay
            for (int i = 0; i < paramIndices.size(); ++i)
            {
                int indexOnTape = paramIndices[i];
                Value& p = tape.getValues()[indexOnTape];
                m[i] = beta1 * m[i] + (1 - beta1) * grad.wrt(p);
                v[i] = beta2 * v[i] + (1 - beta2) * grad.wrt(p) * grad.wrt(p);
                auto m_hat = m[i] / (1 - std::pow(beta1, step + 1));
                auto v_hat = v[i] / (1 - std::pow(beta2, step + 1));

                double delta = lr_t * m_hat / (std::pow(v_hat, 0.5) + eps_adam);
                if (std::abs(delta) > 1e-8)
                {
                    paramCorrCount++;
                    paramCorrTotal += std::abs(delta);
                }
                p.set(p.get() - delta);
                grad.derivs[indexOnTape] = 0.0;
            }

            tape.clearFromIndex(n.getIndex() + 1);

            std::cout << "  Corrected " << paramCorrCount << " params with total value=" << paramCorrTotal << std::endl;
            std::cout << "  Tape size=" << tape.getValues().size() << ", Grad size=" << grad.derivs.size() << std::endl; 
            ++step;
        }
        std::cout << "Training took " << startTimer.get() << " secs." << std::endl;
    }

    void inference(const std::function<Vect(Vect&)>& network, 
                   const std::vector<std::pair<std::vector<double>, int>>& Xs, 
                   Tape& tape)
    {
        utils::Clock startTimer;
        int correct = 0;
        Value sentinel = tape.newValue();
        for (int sampleId = 0; sampleId < Xs.size(); ++sampleId)
        {
            auto xs = Xs[sampleId];
            int labelId = xs.second;
            Vect x(tape, {xs.first.begin(), xs.first.end()});
            auto probs = network(x);

            double maxElem = 0;
            int pred = 0;
            for (int i = 0; i < probs.size(); ++i)
            {
                if (probs[i].get() > maxElem)
                {
                    maxElem = probs[i].get();
                    pred = i;
                }
            }

            //std::cout << "sample id=" << sampleId << " probs=" << probs << " pred=" << pred << " y=" << labelId << '\n';
            if (pred == labelId) correct++;
        }
        tape.clearFromIndex(sentinel.getIndex() + 1);
        std::cout << "correct: " << correct << "/" << Xs.size() << std::endl;
        std::cout << "Inference took " << startTimer.get() << " secs." << std::endl;
    }

}   // namespace mininnet