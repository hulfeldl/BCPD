/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "bcpd.h"

int main()
{

    BCPD<double, 3> bcpd(1.0, 1.0, 0.1, 1.0);
    bcpd.Compute();
    bcpd.Compute();
}