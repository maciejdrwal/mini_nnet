#include "matrix.h"

namespace mininnet
{
    Matrix::Matrix(Tape& tape, size_t m, size_t n) : _tape(tape)
    {
        std::cout << "constructing matrix with " << m << " rows and " << n << " columns\n";
        //static std::random_device rd{};
        static std::mt19937 gen{123};
        static std::normal_distribution<double> distr(0.0, 0.08);
        for (int i = 0; i < m; ++i)
        {
            std::vector<int> row(n);
            std::generate(row.begin(), row.end(), [&](){ return tape.newValue(distr(gen)).getIndex(); });
            _data.emplace_back(std::move(row));
        }
    }

    std::ostream& operator<<(std::ostream &stream, const Matrix &mat)
    {
        stream << std::setprecision(4);
        for (int i = 0; i < mat.numRows(); ++i)
        {
            for (int j = 0; j < mat.numCols(); ++j)
            {
                stream << mat.at(i, j).get() << ' ';
            }
            stream << '\n';
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream &stream, const Vect &vect)
    {
        stream << std::setprecision(4) << '[';
        for (const auto &i : vect._data)
        {
            stream << vect._tape.getValues()[i].get() << ' ';
        }
        stream << ']';
        return stream;
    }

}   // namespace mininnet