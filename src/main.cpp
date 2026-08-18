/**
 * @file main.cpp
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 * @brief 
 * @version 1.1.0
 * @date 2024-2026
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