#include "autograd.h"

#include <cmath>

#include <iostream> // TODO remove

double Grad::wrt(const Value& v) const
{
    return _derivs[v.getIndex()];
}

Value& Tape::newValue(double value, std::array<int, 2> children, std::array<double, 2> lgs)
{
    int nextIndex = _values.size();
    return _values.emplace_back(Value(this, nextIndex, value, children, lgs));
}

Value& Value::operator+=(const Value& other) const
{
    return _tape->newValue(_data + other._data, { _index, other._index }, { 1.0, 1.0 } );
}

Value Value::operator+(const Value& other) const
{
    return *this += other;
    //return _tape->newValue(_data + other._data, { _index, other._index }, { 1.0, 1.0 });
}

Value Value::operator*(const Value& other) const
{
    return _tape->newValue(_data * other._data, { _index, other._index }, { other._data, _data });
}

Value Value::pow(double p) const
{
    return _tape->newValue(std::pow(_data, p), { _index }, { p * std::pow(_data, p-1) } );
}

Value Value::log() const
{
    return _tape->newValue(std::log(_data), { _index }, { 1.0 / _data } );
}

Value Value::exp() const
{
    return _tape->newValue(std::exp(_data), { _index }, { std::exp(_data) });
}

Value Value::relu() const
{
    return _tape->newValue(std::max(0.0, _data), { _index }, { static_cast<double>(_data > 0.0) });
}

Value Value::neg() const
{
    // ???
    return *this * _tape->newValue(-1);
}

Value Value::operator-(Value& other) const
{
    return _tape->newValue(_data - other._data, { _index, other._index }, { _data, -other._data});
}

Value Value::operator/(Value& other) const
{
    return _tape->newValue(_data * std::pow(other._data, -1.0), { _index, other._index }, { _data, -std::pow(_data, -2.0)});
}

Value Value::sin() const
{
    return _tape->newValue(std::sin(_data), { _index }, { std::cos(_data) });
}

Grad Value::backward()
{
    auto& values = _tape->getValues();
    std::vector<double> derivs(values.size(), 0.0);
    derivs[_index] = 1.0;

    for (int i = values.size() - 1; i >= 0; --i)
    {
        Value& val = values[i];
        double deriv = derivs[i];

        // std::cout << "  val=" << val.get() << " deriv=" << deriv << " ch1=" << val._children[0] << " ch2=" << val._children[1]
        //     << " lg0=" << val._localGrads[0] << " lg1=" << val._localGrads[1] << std::endl;

        derivs[val._children[0]] += val._localGrads[0] * deriv;
        derivs[val._children[1]] += val._localGrads[1] * deriv;
    }

    return { derivs };
}
