#include "autograd.h"

#include <cmath>

namespace mininnet
{

double Grad::wrt(const Value& v) const
{
    return derivs[v.getIndex()];
}

Value& Tape::newValue(double value, std::array<int, 2> children, std::array<double, 2> lgs)
{
    int nextIndex = _values.size();
    return _values.emplace_back(Value(*this, nextIndex, value, children, lgs));
}

Value& Value::operator+=(const Value& other)
{
    *this = _tape.newValue(_data + other._data, { _index, other._index }, { 1.0, 1.0 } );
    return *this;
}

Value Value::operator+(const Value& other) const
{
    return _tape.newValue(_data + other._data, { _index, other._index }, { 1.0, 1.0 });
}

Value Value::operator*(const Value& other) const
{
    return _tape.newValue(_data * other._data, { _index, other._index }, { other._data, _data });
}

Value Value::pow(double p) const
{
    return _tape.newValue(std::pow(_data, p), { _index, 0 }, { p * std::pow(_data, p-1), 0 } );
}

Value Value::log() const
{
    return _tape.newValue(std::log(_data), { _index, 0 }, { 1.0 / _data, 0 } );
}

Value Value::exp() const
{
    return _tape.newValue(std::exp(_data), { _index, 0 }, { std::exp(_data), 0 });
}

Value Value::relu() const
{
    return _tape.newValue(std::max(0.0, _data), { _index, 0 }, { static_cast<double>(_data > 0.0), 0 });
}

Value Value::neg() const
{
    return *this * _tape.newValue(-1);
}

Value& Value::operator-=(const Value& other)
{
    *this = *this + other.neg();
    return *this;
}

Value Value::operator-(Value& other) const
{
    return *this + other.neg();
}

Value Value::operator/(Value& other) const
{
    return *this * other.pow(-1);
}

Value Value::sin() const
{
    return _tape.newValue(std::sin(_data), { _index, 0 }, { std::cos(_data), 0 });
}

Value& Value::operator=(const Value& val)
{
    _data = val._data;
    _index = val._index;
    _children = val._children;
    _localGrads = val._localGrads;
    return *this;
}

Grad Value::backprop() const
{
    const auto& values = _tape.getValues();
    std::deque<double> derivs(values.size(), 0.0);
    derivs[_index] = 1.0;

    for (int i = values.size() - 1; i >= 0; --i)
    {
        const Value& val = values[i];
        double deriv = derivs[i];

        derivs[val._children[0]] += val._localGrads[0] * deriv;
        derivs[val._children[1]] += val._localGrads[1] * deriv;
    }

    return { derivs };
}

} // namespace mininnet
