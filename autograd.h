#include <array>
#include <vector>


class Value;

struct Grad
{
    std::vector<double> _derivs;

    double wrt(const Value& v) const;
};

class Tape;

class Value
{
public:
    Value& operator+=(const Value& other) const;
    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value pow(double p) const;
    Value log() const;
    Value exp() const;
    Value relu() const;
    Value neg() const;
    Value operator-(Value& other) const;
    Value operator/(Value& other) const;
    Value sin() const;

    // Value operator/(double other)
    // {
    //     return *this * Value(other).pow(-1.0);
    // }

    double get() const { return _data; }
    int getIndex() const { return _index; }
    Grad backward();

    friend class Tape;

protected:
    Tape* _tape;
    int _index;
    double _data;
    std::array<int, 2> _children = { 0, 0 };
    std::array<double, 2> _localGrads = { 0, 0 };

    Value(Tape* tape, int index, double value = 0, std::array<int, 2> children = {}, std::array<double, 2> lgs = {})
    : _tape(tape), _index(index), _data(value), _children(std::move(children)), _localGrads(std::move(lgs)) {}
};


class Tape
{
public:
    Tape() = default;

    Value& newValue(double value = 0, std::array<int, 2> children = {}, std::array<double, 2> lgs = {});
    
    std::vector<Value>& getValues() { return _values; }

protected:
    std::vector<Value> _values;
};

