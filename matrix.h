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
    Vect(Tape& tape, size_t m, double d = 0);
    Vect(Tape& tape, const std::vector<int>& indices) : _tape(tape), _data(indices) {}
    Vect(Tape& tape, const std::initializer_list<double> ilist);

    Value& operator[](int i) const;
    Vect& operator=(const Vect& other);

    Value& at(int i) const;
    Vect slice(int from, int to) const;

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

    const Value& at(int i, int j) const;

    // get i-th row
    Vect at(int i) const;

    int numRows() const { return _data.size(); }
    int numCols() const { return _data.at(0).size(); }

    std::vector<int> getIndices() const;

    void serialize(const std::string& filename);
    void deserialize(const std::string& filename);

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& mat);

protected:
    Tape& _tape;
    std::vector<std::vector<int>> _data;
};

std::ostream& operator<<(std::ostream& stream, const Matrix& mat);
std::ostream& operator<<(std::ostream& stream, const Vect& vect);

} // namespace mininnet