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
        // Vect xResidual = x;
        // x = rmsnorm(x);
        // x = linear(x, W1);
        // for (int i = 0; i < x.size(); ++i)
        // {
        //     x[i] = x[i].relu(); 
        // }
        // x = linear(x, W2);
        // for (int i = 0; i < x.size(); ++i)
        // {
        //     x[i] += xResidual[i];
        // }
        return softmax(x);
    }

    void training(const std::function<Vect(Vect&)>& network, 
                  const std::vector<std::pair<std::vector<double>, int>>& Xs, 
                  Tape& tape,             
                  const std::vector<int>& paramIndices)
    {
        int step = 0;
        int batchSize = 150;
        int numSteps = 1;
        double learningRate = 0.01;
        double beta1 = 0.85;
        double beta2 = 0.99;
        double eps_adam = 1e-8;
        std::vector<double> m(paramIndices.size(), 0.0);  // first moment buffer
        std::vector<double> v(paramIndices.size(), 0.0);  // second moment buffer

        std::cout <<  Xs.size() << std::endl;

        Value n = tape.newValue(batchSize);
        Value sumLoss = tape.newValue(0.0);
        while (step < numSteps)
        {
            std::cout << "Training step: " << step << std::endl;
            sumLoss.set(0.0);
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

            std::cout << "Loss before backward=" << loss.get() << " index=" << loss.getIndex() << std::endl;
            
            Grad grad = loss.backward();

            double lr_t = learningRate * (1 - step / numSteps); // linear learning rate decay
            for (int i = 0; i < paramIndices.size(); ++i)
            {
                int indexOnTape = paramIndices[i];
                Value& p = tape.getValues()[indexOnTape];
                m[i] = beta1 * m[i] + (1 - beta1) * grad.wrt(p);
                v[i] = beta2 * v[i] + (1 - beta2) * grad.wrt(p) * grad.wrt(p);
                auto m_hat = m[i] / (1 - std::pow(beta1, step + 1));
                auto v_hat = v[i] / (1 - std::pow(beta2, step + 1));

                std::cout << "i=" << i << " tape index=" << indexOnTape << " p=" << p.get() << " loss/dp=" << grad.wrt(p) << std::endl;
                std::cout << "m_hat=" << m_hat << " v_hat=" << v_hat << " m[i]=" << m[i] << " v[i]=" << v[i] << std::endl;

                double acor =  lr_t * m_hat / (std::pow(v_hat, 0.5) + eps_adam);
                std::cout << "adam correction=" << acor << std::endl;
                p.set(p.get() - acor);
                std::cout << "new p=" << p.get() << std::endl;
                grad.derivs[indexOnTape] = 0.0;
            }
            
            std::cout << "loss=" << loss.get() << std::endl;

            ++step;
        }
    }

    void inference(const std::function<Vect(Vect&)>& network, 
                   const std::vector<std::pair<std::vector<double>, int>>& Xs, 
                   Tape& tape)
    {
        int correct = 0;
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

            std::cout << "sample id=" << sampleId << " probs=" << probs << " pred=" << pred << " y=" << labelId << '\n';
            if (pred == labelId) correct++;
        }
        std::cout << "correct: " << correct << "/" << Xs.size() << std::endl; 
    }

}   // namespace mininnet