#include "autograd.h"
#include <iostream>

int main()
{
    Tape tape;

    Value x = tape.newValue(0.5);
    Value y = tape.newValue(4.2);
    Value z = x * y + x.sin();

    auto grad = z.backward();

    std::cout << "z = " << z.get() 
              << "\ndz/dx = " << grad.wrt(x)
              << "\ndz/dy = " << grad.wrt(y) << std::endl;
    
}
