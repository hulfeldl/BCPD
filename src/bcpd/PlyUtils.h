/**
 * @file PlyUtils.h
 * @brief Utilities for writing PLY files.
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>

/**
 * @brief Writes points to a binary PLY file.
 *
 * @tparam FloatType Floating point type.
 * @tparam Dim Dimensionality of the points.
 *
 * @param points Points to write.
 * @param path Output file path (without extension); "-binary.ply" is appended.
 */
template <typename FloatType, uint32_t Dim>
void writePly(const std::vector<Eigen::Vector<FloatType, Dim>>& points,
              const std::filesystem::path& path);