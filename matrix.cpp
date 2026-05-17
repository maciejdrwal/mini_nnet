#include "matrix.h"

#include <sstream>
#include <fstream>

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

    void Matrix::serialize(const std::string& filename)
    {
        const std::size_t maxPrecision = std::numeric_limits<double>::digits;
        std::ofstream stream(filename);

        for (int row = 0; row < _data.size(); ++row)
        {
            const auto& rowData = _data[row];
            for (int col = 0; col < rowData.size(); ++col)
            {
                const double number = _tape.getValues()[rowData.at(col)].get();
                uint64_t ismall;
                std::memcpy((void *)&ismall, (void *)&number, sizeof(number));
                stream.precision(maxPrecision);
                stream << ismall << ' ';
            }
            stream << '\n';
        }
    }
    
    void Matrix::deserialize(const std::string& filename)
    {
        _data.clear();
        std::ifstream infile(filename);
        for (std::string line; std::getline(infile, line, '\n'); )
        {
            std::istringstream ss(line);
            for (std::string term; std::getline(ss, term, ' '); )
            {            
                uint64_t ismall = std::stoull(term);
                double fsmall;
                std::memcpy((void *)&fsmall, (void *)&ismall, sizeof(fsmall));
                std::cout << "de-serialized: " << fsmall << std::endl;
            }
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