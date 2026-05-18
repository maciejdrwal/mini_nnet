#include "network.h"
#include "mem_usage.h"
#include "utils.h"

#include <algorithm>

namespace mininnet
{
    Vect NeuralNetwork::linear(const Vect& x, const Matrix& W) const
    {
        if (x.size() != W.numCols())
        {
            throw std::runtime_error("linear: invalid dimensions: dim(x)=" + std::to_string(x.size()) +
                                     " cols="  + std::to_string(W.numCols()) + " rows=" + std::to_string(W.numRows()));
        }

        Vect result(_tape, W.numRows());
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

    Vect NeuralNetwork::rmsnorm(const Vect& x) const
    {
        Value ms = _tape.newValue(0);
        Value eps = _tape.newValue(1e-5);
        Value d = _tape.newValue(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            ms += x[i] * x[i];
        }
        ms = ((ms + eps) / d).pow(0.5);

        Vect result(_tape, x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = x[i] * ms; 
        }
        return result;
    }

    Vect NeuralNetwork::softmax(const Vect& x) const
    {
        double maxElem = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < x.size(); ++i)
        {
            maxElem = std::max(x[i].get(), maxElem);
        }
        Value maxVal = _tape.newValue(maxElem);
        Value total = _tape.newValue(0.0);
        Vect exps(_tape, x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            exps[i] = (x[i] - maxVal).exp();
            total += exps[i];
        }
        Vect result(_tape, x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            result[i] = exps[i] / total;
        }
        return result;
    }

    Vect NeuralNetwork::MLPBlock(Vect& x, const Matrix& W1, const Matrix& W2) const
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
        return x;
    }

    Vect NeuralNetwork::AttentionBlock(Vect& x, const Matrix& Wq, const Matrix& Wk, const Matrix& Wv, const Matrix& Wo,
                        std::vector<Vect>& keys, std::vector<Vect>& values, int numHeads) const
    {
        const int dimEmbed = x.size();
        const int dimHead = dimEmbed / numHeads;
        Value dimHeadSqrt = _tape.newValue(std::pow(dimHead, 0.5));

        Vect xRes = x;
        x = rmsnorm(x);
        Vect q = linear(x, Wq);
        Vect k = linear(x, Wk);
        Vect v = linear(x, Wv);
        keys.push_back(k);
        values.push_back(v);

        Vect xAttn(_tape, dimEmbed);

        for (int h = 0; h < numHeads; ++h)
        {
            int hs = h * dimHead;
            Vect q_h = q.slice(hs, hs + dimHead);            
            std::vector<Vect> k_h, v_h;
            for (int i = 0; i < keys.size(); ++i)
            {
                k_h.push_back(keys[i].slice(hs, hs + dimHead));
            }
            for (int i = 0; i < values.size(); ++i)
            {
                v_h.push_back(values[i].slice(hs, hs + dimHead));
            }

            Vect attnLogits(_tape, k_h.size());
            for (int t = 0; t < k_h.size(); ++t)
            {
                for (int j = 0; j < dimHead; ++j)
                {
                    attnLogits[t] += q_h[j] * k_h[t][j];
                }
                attnLogits[t] = attnLogits[t] / dimHeadSqrt;
            }
            Vect attnWeights = softmax(attnLogits);
            Vect headOut(_tape, dimHead);
            for (int j = 0; j < dimHead; ++j)
            {
                for (int t = 0; t < v_h.size(); ++t)
                {
                    headOut[j] += attnWeights[t] * v_h[t][j];
                }
                xAttn[hs + j] = headOut[j];
            }
        }

        x = linear(xAttn, Wo);
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] += xRes[i];
        }
        return x;
    }

    void NeuralNetwork::training(int numSteps)
    {
        utils::Clock startTimer;
        int step = 0;
        double learningRate = 0.01;
        double beta1 = 0.85;
        double beta2 = 0.99;
        double eps_adam = 1e-8;
        std::vector<double> m(_paramIndices.size(), 0.0);  // first moment buffer
        std::vector<double> v(_paramIndices.size(), 0.0);  // second moment buffer
        int startPos = _tape.size();

        while (step < numSteps)
        {
            std::cout << "Training step: " << step << std::endl;

            Value loss = processBatch(step);

            std::cout << "Loss=" << loss.get() << std::endl;
            if (std::isnan(loss.get()))
            {
                throw std::runtime_error("NaN encountered during step=" + std::to_string(step));
            }
            
            Grad grad = loss.backprop();

            int paramCorrCount = 0;
            double paramCorrTotal = 0;

            double lr_t = learningRate * (1 - step / numSteps); // linear learning rate decay
            for (int i = 0; i < _paramIndices.size(); ++i)
            {
                int indexOnTape = _paramIndices[i];
                Value& p = _tape.getValues()[indexOnTape];
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

            _tape.clearFromIndex(startPos);

            std::cout << "  Corrected " << paramCorrCount << " params with total value=" << paramCorrTotal << std::endl;
            std::cout << "  Tape size=" << _tape.getValues().size() << ", Grad size=" << grad.derivs.size() << std::endl; 
            if (step % 10 == 0) std::cout << mem_usage::mem_usage_summary() << std::endl;
            ++step;
        }
        std::cout << "Training took " << startTimer.get() << " secs." << std::endl;
    }


    Value Classifier::processBatchSupervised(int step, int batchSize) const
    {
        Value n = _tape.newValue(batchSize);
        Value sumLoss = _tape.newValue(0.0);
        for (int j = 0; j < batchSize; ++j)
        {
            int sampleId = (step * batchSize + j) % _Xs.size();
            const auto &xs = _Xs[sampleId];
            int labelId = xs.second;
            Vect x(_tape, {xs.first.begin(), xs.first.end()});
            auto probs = evaluate(x);
            sumLoss -= probs[labelId].log();
        }
        Value loss = sumLoss / n;
        return loss;
    }

    void Classifier::inference(const LabeledData& Xs) const
    {
        utils::Clock startTimer;
        int correct = 0;
        for (int sampleId = 0; sampleId < Xs.size(); ++sampleId)
        {
            const auto& xs = Xs[sampleId];
            int labelId = xs.second;
            Vect x(_tape, {xs.first.begin(), xs.first.end()});
            const Vect probs = evaluate(x);

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
            _tape.clearFromIndex(x[0].getIndex());

            if (pred == labelId) correct++;
        }
        
        std::cout << "Correct predictions: " << correct << "/" << Xs.size() << std::endl;
        std::cout << "Inference took " << startTimer.get() << " secs." << std::endl;
        std::cout << mem_usage::mem_usage_summary() << std::endl;
    }

}   // namespace mininnet
