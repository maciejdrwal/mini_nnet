#pragma once

#include "matrix.h"

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
                result[i] += wi[j] * x[i];
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
            if (x[i].get() > maxElem) maxElem = x[i].get();
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
                  const std::vector<std::vector<double>>& Xs, 
                  const std::vector<double>& ys,
                  Tape& tape,             
                  std::vector<int>& paramIndices)
    {
        int step = 0;
        int batchSize = 100;
        int numSteps = 1000;
        double learningRate = 0.01;
        double beta1 = 0.85;
        double beta2 = 0.99;
        double eps_adam = 1e-8;
        std::vector<double> m(tape.getValues().size(), 0.0);  // first moment buffer
        std::vector<double> v(tape.getValues().size(), 0.0);  // second moment buffer

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
                int labelId = std::clamp(static_cast<int>(ys[sampleId]), 0, 9);
                Vect x(tape, {xs.begin(), xs.end()});
                auto probs = network(x);

                std::cout << "sample:" << sampleId << " label:" << labelId << std::endl;

                sumLoss += probs[labelId].log().neg();
            }
        
            std::cout << "batch done!" << std::endl;

            auto loss = sumLoss / n;
            
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

                p.set(p.get() - lr_t * m_hat / (std::pow(v_hat, 0.5) + eps_adam));
                grad.derivs[i] = 0.0;
            }
            
            std::cout << "loss=" << loss.get() << std::endl;

            ++step;
        }
    }

}   // namespace mininnet