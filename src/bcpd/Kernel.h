/**
 * @file Kernel.h
 * @brief Base class for kernel functions used in BCPD
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <cstdint>

template <class FloatType, uint32_t dim> class Kernel
{
public:
    using VectorType = Eigen::Vector<FloatType, dim>;
    virtual ~Kernel() = default;
    virtual FloatType compute(const VectorType& x, const VectorType& y) const = 0;
};