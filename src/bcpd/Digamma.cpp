/**
 * @file Digamma.cpp
 * @brief Digamma function (derivative of the log-gamma function).
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#include "Digamma.h"

#include <cmath>
#include <limits>
#include <numbers>

namespace
{
    /**
     * @brief Computes the digamma function on x >= 0.5 via the asymptotic series, after the
     * recurrence relation psi(x) = psi(x+1) - 1/x has shifted x above 5.
     *
     * @param x Input value; must be >= 0.5.
     *
     * @return Digamma of x.
     */
    [[nodiscard]] double digammaSeries(double x) noexcept
    {
        double r = 0.0;
        while (x <= 5.0)
        {
            r -= 1.0 / x;
            x += 1.0;
        }

        const double f = 1.0 / (x * x);
        const double t =
            f *
            (-1.0 / 12.0 +
             f * (1.0 / 120.0 +
                  f * (-1.0 / 252.0 +
                       f * (1.0 / 240.0 +
                            f * (-1.0 / 132.0 + f * (691.0 / 32760.0 +
                                                     f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));

        return r + std::log(x) - 0.5 / x + t;
    }
}  // namespace

template <typename FloatType>
FloatType digamma(FloatType x_in) noexcept
{
    const double x = static_cast<double>(x_in);

    if (x < 0.5)
    {
        const double piX = std::numbers::pi_v<double> * x;
        const double s = std::sin(piX);
        if (s == 0.0)
        {
            return std::numeric_limits<FloatType>::quiet_NaN();
        }
        return static_cast<FloatType>(digammaSeries(1.0 - x) -
                                      std::numbers::pi_v<double> * std::cos(piX) / s);
    }

    return static_cast<FloatType>(digammaSeries(x));
}

template double digamma<double>(double x_in);
