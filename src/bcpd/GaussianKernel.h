/**
 * @file GaussianKernel.h
 * @brief Gaussian kernel implementation for BCPD
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#pragma once

#include "Kernel.h"

#include <cmath>
#include <cstdint>

template <typename FloatType, uint32_t Dim>
class GaussKernel : public Kernel<FloatType, Dim>
{
public:
    explicit GaussKernel(FloatType beta) noexcept : h(beta) {}

    [[nodiscard]] FloatType compute(
        const typename Kernel<FloatType, Dim>::VectorType& x,
        const typename Kernel<FloatType, Dim>::VectorType& y) const override
    {
        return std::exp(-(x - y).dot(x - y) / (2.0 * h * h));
    }

private:
    FloatType h = 0.5;
};
