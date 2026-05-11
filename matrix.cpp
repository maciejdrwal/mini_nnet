#include "matrix.h"

namespace mininnet
{
    std::ostream &operator<<(std::ostream &stream, const Matrix &mat)
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

    std::ostream &operator<<(std::ostream &stream, const Vect &vect)
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