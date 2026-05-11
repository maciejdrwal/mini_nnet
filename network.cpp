#include "network.h"
#include "mem_usage.h"
#include "utils.h"

#include <algorithm>

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

    Vect AttentionBlock(Vect& x, Matrix& Wq, Matrix& Wk, Matrix& Wv, Matrix& Wo,
        std::vector<Vect>& keys, std::vector<Vect>& values, 
        int numHeads)
    {
        const int dimEmbed = x.size();
        const int headDim = dimEmbed / numHeads;
        Value headDimSqrt = x[0].getTape().newValue(std::pow(headDim, 0.5));

        Vect xResidual = x;
        x = rmsnorm(x);
        auto q = linear(x, Wq);
        auto k = linear(x, Wk);
        auto v = linear(x, Wv);
        keys.push_back(k);
        values.push_back(v);
        Vect xAttn(x[0].getTape(), dimEmbed /* = numHeads * headDim */);

        // note: can be computed in parallel
        for (int h = 0; h < numHeads; ++h)
        {
            int hs = h * headDim;
            Vect q_h = q.slice(hs, hs + headDim);
            std::vector<Vect> k_h, v_h;
            for (int i = 0; i < keys.size(); ++i)
            {
                k_h.emplace_back(keys[i].slice(hs, hs + headDim));
            }
            for (int i = 0; i < values.size(); ++i)
            {
                v_h.emplace_back(values[i].slice(hs, hs + headDim));
            }

            Vect attnLogits(x[0].getTape(), k_h.size());
            for (int i = 0; i < k_h.size(); ++i)
            {
                Value s = x[0].getTape().newValue(0.0);
                for (int j = 0; j < headDim; ++j)
                {
                    s += q_h[j] * k_h[i][j];
                }
                attnLogits[i] = s / headDimSqrt;
            }
            Vect attnWeights = softmax(attnLogits);

            Vect headOut(x[0].getTape(), headDim, 0.0);
            for (int j = 0; j < headDim; ++j)
            {
                for (int i = 0; i < v_h.size(); ++i)
                {
                    headOut[j] += attnWeights[i] * v_h[i][j];
                }
                xAttn[h * headDim + j] = headOut[j];
            }
        }

        x = linear(xAttn, Wo);
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] += xResidual[i];
        }
        return x;
    }

    void training(/*const std::function<Vect(Vect&)>& network, 
                  const LabeledData& Xs,*/
                  const std::function<Value(Tape&, int)>& processBatch, 
                  Tape& tape,             
                  const std::vector<int>& paramIndices, 
                  int numSteps)
    {
        utils::Clock startTimer;
        int step = 0;
        double learningRate = 0.01;
        double beta1 = 0.85;
        double beta2 = 0.99;
        double eps_adam = 1e-8;
        std::vector<double> m(paramIndices.size(), 0.0);  // first moment buffer
        std::vector<double> v(paramIndices.size(), 0.0);  // second moment buffer
        int startPos = tape.size();

        while (step < numSteps)
        {
            std::cout << "Training step: " << step << std::endl;

            Value loss = processBatch(tape, step);
            
            std::cout << "Loss=" << loss.get() << std::endl;

            Grad grad = loss.backprop();

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

            tape.clearFromIndex(startPos);

            std::cout << "  Corrected " << paramCorrCount << " params with total value=" << paramCorrTotal << std::endl;
            std::cout << "  Tape size=" << tape.getValues().size() << ", Grad size=" << grad.derivs.size() << std::endl; 
            std::cout << mem_usage::mem_usage_summary() << std::endl;
            ++step;
        }
        std::cout << "Training took " << startTimer.get() << " secs." << std::endl;
    }

    void inference(const std::function<Vect(Vect&)>& network, 
                   const LabeledData& Xs, 
                   Tape& tape)
    {
        utils::Clock startTimer;
        int correct = 0;
        for (int sampleId = 0; sampleId < Xs.size(); ++sampleId)
        {
            const auto& xs = Xs[sampleId];
            int labelId = xs.second;
            Vect x(tape, {xs.first.begin(), xs.first.end()});
            const Vect probs = network(x);

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
            tape.clearFromIndex(x[0].getIndex());

            //std::cout << "sample id=" << sampleId << " probs=" << probs << " pred=" << pred << " y=" << labelId << '\n';
            if (pred == labelId) correct++;
        }
        
        std::cout << "Correct predictions: " << correct << "/" << Xs.size() << std::endl;
        std::cout << "Inference took " << startTimer.get() << " secs." << std::endl;
        std::cout << mem_usage::mem_usage_summary() << std::endl;
    }

}   // namespace mininnet
