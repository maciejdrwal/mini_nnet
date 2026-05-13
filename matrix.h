#pragma once

#include "autograd.h"

#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

namespace mininnet
{

struct Vect
{
    Vect(Tape& tape, size_t m, double d = 0) : _tape(tape)
    {
        _data.reserve(m);
        for (int i = 0; i < m; ++i)
        {
            _data.push_back(tape.newValue(d).getIndex());
        }
    }

    Vect(Tape& tape, const std::vector<int>& indices) : _tape(tape), _data(indices) {}

    Vect(Tape& tape, const std::initializer_list<double> ilist) : _tape(tape)
    {
        _data.reserve(ilist.size());
        for (const auto d : ilist)
        {
            _data.push_back(tape.newValue(d).getIndex());
        }
    }

    Value& at(int i) const
    {
        if (i > _data.size() || i < 0)
        {
            throw std::runtime_error("Invalid vector subscript");
        }
        return operator[](i);
    }

    Value& operator[](int i) const
    {
        return _tape.getValues()[_data[i]];
    }

    Vect slice(int from, int to) const
    {
        if (to <= from || to > _data.size())
        {
            throw std::runtime_error("Invalid slice indexing");
        }
        Vect result(_tape, to - from);
        for (int i = 0; i < to - from; ++i)
        {
            result._data[i] = _data[from + i];
        }
        return result;
    }

    Vect& operator=(const Vect& other)
    {
        _data = other._data;
        return *this;
    }

    size_t size() const { return _data.size(); }

    friend std::ostream& operator<<(std::ostream& stream, const Vect& vect);

protected:
    Tape& _tape;
    std::vector<int> _data;
};


class Matrix
{
public:
    // m - rows, n - columns
    Matrix(Tape& tape, size_t m, size_t n);

    const Value& at(int i, int j) const
    {
        if (i >= numRows() || j >= numCols())
        {
            throw std::runtime_error("Invalid matrix subscripts");
        }
        return _tape.getValues()[_data[i][j]];
    }

    // get i-th row
    Vect at(int i)
    {
        std::vector<int>& indices = _data[i];
        return Vect(_tape, indices);
    }

    int numRows() const { return _data.size(); }
    int numCols() const { return _data.at(0).size(); }

    std::vector<int> getIndices() const 
    {
        std::vector<int> result;
        result.reserve(numRows() * numCols()); 
        for (const auto& row : _data)
        {
            result.insert(result.end(), row.begin(), row.end());
        }
        return result; 
    }

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& mat);

protected:
    Tape& _tape;
    std::vector<std::vector<int>> _data;
};

std::ostream& operator<<(std::ostream& stream, const Matrix& mat);
std::ostream& operator<<(std::ostream& stream, const Vect& vect);

} // namespace mininnet