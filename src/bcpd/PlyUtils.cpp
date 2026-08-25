/**
 * @file PlyUtils.cpp
 * @brief Utilities for writing PLY files.
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#include "PlyUtils.h"

#include <fstream>
#include <tinyply.h>

template <typename FloatType, uint32_t Dim>
void writePly(const std::vector<Eigen::Vector<FloatType, Dim>>& points,
              const std::filesystem::path& path)
{
    struct Vertex
    {
        double x, y, z;
    };

    std::vector<Vertex> pointsOut;
    pointsOut.reserve(points.size());

    if constexpr (Dim == 3)
    {
        for (const auto& point : points)
        {
            pointsOut.push_back({point[0], point[1], point[2]});
        }
    }
    else if constexpr (Dim == 2)
    {
        for (const auto& point : points)
        {
            pointsOut.push_back({point[0], point[1], FloatType(0.0)});
        }
    }

    tinyply::PlyFile file;
    file.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
                                   points.size(), reinterpret_cast<uint8_t*>(pointsOut.data()),
                                   tinyply::Type::INVALID, 0);

    std::filebuf fbBinary;
    std::string filename = path.string(); // + "-binary.ply";
    fbBinary.open(filename, std::ios::out | std::ios::binary);
    std::ostream outstream(&fbBinary);

    if (outstream.fail())
    {
        throw std::runtime_error("Failed to open " + filename);
    }
    file.write(outstream, true);
}

template void writePly<double, 2u>(const std::vector<Eigen::Vector<double, 2u>>& points,
                                   const std::filesystem::path& path);
template void writePly<double, 3u>(const std::vector<Eigen::Vector<double, 3u>>& points,
                                   const std::filesystem::path& path);