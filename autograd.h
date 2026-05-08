#pragma once

#include <array>
#include <deque>

namespace mininnet
{
class Value;

struct Grad
{
    std::deque<double> derivs;

    double wrt(const Value& v) const;
};

class Tape;

class Value
{
public:
    Value& operator+=(const Value& other);
    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value pow(double p) const;
    Value log() const;
    Value exp() const;
    Value relu() const;
    Value neg() const;
    Value operator-(Value& other) const;
    Value& operator-=(const Value& other);
    Value operator/(Value& other) const;
    Value sin() const;

    Value& operator=(const Value& val);

    double get() const { return _data; }
    void set(double v) { _data = v; _children = {0, 0}; _localGrads = {0, 0}; }
    int getIndex() const { return _index; }
    Tape& getTape() { return _tape; }
    Grad backward();

    int getChild(int i) const { return _children[i]; }
    double getGrad(int i) const { return _localGrads[i]; }

    friend class Tape;

protected:
    Tape& _tape;
    int _index;
    double _data;
    std::array<int, 2> _children = { 0, 0 };
    std::array<double, 2> _localGrads = { 0, 0 };

    Value(Tape& tape, int index, double value, std::array<int, 2> children, std::array<double, 2> lgs)
    : _tape(tape), _index(index), _data(value), _children(std::move(children)), _localGrads(std::move(lgs)) {}
};

class Tape
{
public:
    Tape() = default;

    Value& newValue(double value = 0, std::array<int, 2> children = {0, 0}, std::array<double, 2> lgs = {0, 0});
    std::deque<Value>& getValues() { return _values; }
    void clearFromIndex(int index)
    {
        _values.erase(_values.begin() + index, _values.end());
    }

protected:
    std::deque<Value> _values;
};

} // namespace mininnet
