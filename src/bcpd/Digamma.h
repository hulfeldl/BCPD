/**
 * @file Digamma.h
 * @brief Digamma function (derivative of the log-gamma function).
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#pragma once

/**
 * @brief Computes the digamma function (derivative of the log-gamma function).
 *
 * @tparam FloatType Floating point type.
 *
 * @param x_in Input value.
 *
 * @return Digamma of x_in, or NaN at a pole (x_in a non-positive integer).
 *
 * @note x_in is expected to be positive (callers typically pass expectation-like quantities
 * such as nu/N_hat); a negative x_in signals upstream numerical corruption, but is still
 * handled safely here via the reflection formula rather than looping, since the underlying
 * recurrence relation would otherwise need ~|x_in| iterations, or never terminate once |x_in|
 * is large enough that incrementing by 1.0 can no longer change it in floating-point.
 */
template <typename FloatType>
[[nodiscard]] FloatType digamma(FloatType x_in) noexcept;
