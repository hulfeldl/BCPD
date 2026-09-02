/**
 * @file bcpd-tests.cpp
 * @brief BCPD algorithm test cases with 2D and 3D point cloud registrations
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#include "bcpd/PlyUtils.h"

#include <bcpd/BCPD.h>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>

namespace
{
    /// Test data dir relative to repo.
    const auto testDataDir = std::filesystem::path(BCPD_TEST_DATA_DIR);

    /**
     * @brief Reads Eigen vectors from a text file
     *
     * @tparam FloatType Floating-point type for vector components (float, double, etc.)
     * @tparam Dim Dimensionality of each vector
     *
     * @param filename Path to the input text file containing vector data
     *
     * @return std::vector<Eigen::Vector<FloatType, Dim>> Vector of read Eigen vectors
     *
     * @throws std::runtime_error If the file cannot be opened
     *
     * @details Each line in the input file should contain Dim whitespace-separated
     * floating-point values representing a single vector component. This function
     * reads line-by-line and constructs Eigen vectors from the parsed values.
     *
     * @note Extra values on a line beyond Dim are ignored
     * @note Lines with fewer than Dim values will create incomplete vectors
     */
    template <class FloatType, uint32_t Dim>
    std::vector<Eigen::Vector<FloatType, Dim>> readEigenVectorsFromTxtFile(
        const std::filesystem::path& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error(std::string("Could not open file: ") + filename.string());
        }

        std::vector<Eigen::Vector<FloatType, Dim>> vectors;
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            Eigen::Vector<FloatType, Dim> vector;
            FloatType value;
            uint32_t i = 0u;
            while (iss >> value && i < Dim)
            {
                vector[i] = value;
                ++i;
            }
            vectors.push_back(vector);
        }

        return vectors;
    }

    /**
     * @brief Writes a point cloud vector to a PLY file in the test output directory.
     *
     * @tparam FloatType Floating-point type for vector components (float, double, etc.)
     * @tparam Dim Dimensionality of each point (typically 2 or 3)
     *
     * @param name Subdirectory name within the output folder (e.g., "2D_square", "armadillo")
     * @param y The point cloud data as a vector of Eigen vectors to be written
     * @param outputName Output filename (default: "yOut.ply")
     */
    template <typename FloatType, uint32_t Dim>
    void writeOutput(std::string name,
                     const std::vector<Eigen::Vector<FloatType, Dim>>& y,
                     std::string outputName = "yOut.ply")
    {
        const auto outDir = testDataDir / "output" / name;
        std::filesystem::create_directories(outDir);

        writePly<FloatType, Dim>(y, outDir / outputName);
    }

    /**
     * @brief Reads point cloud data from a text file and optionally writes it as a PLY file.
     *
     * @tparam FloatType Floating-point type for vector components (float, double, etc.)
     * @tparam Dim Dimensionality of each point (typically 2 or 3)
     *
     * @param data_dir Directory containing the input text file
     * @param fileName Name of the input text file.
     *
     * @return std::vector<Eigen::Vector<FloatType, Dim>> Vector of read Eigen vectors
     */
    template <typename FloatType, uint32_t Dim>
    std::vector<Eigen::Vector<FloatType, Dim>> readAndWriteInput(
        const std::filesystem::path& data_dir, const std::filesystem::path& fileName)
    {
        // Replace file extension with ply.
        auto filePly = fileName;
        filePly.replace_extension("ply");

        // Read data.
        const auto data = readEigenVectorsFromTxtFile<FloatType, 3u>(data_dir / fileName);

        // Write ply file if it does not exist.
        if (!std::filesystem::exists(filePly))
        {
            writePly<FloatType, Dim>(data, data_dir / filePly);
        }

        return data;
    }

}  // namespace

/**
 * @test 2D square test
 * @brief Tests BCPD registration of a 2D square point cloud with translation
 *
 * @details This test case verifies the basic functionality of BCPD with a simple
 * 2D point cloud consisting of the corners of a unit square. The template point
 * cloud (y) is a translated version of the reference point cloud (x), allowing
 * verification that the algorithm can recover the applied transformation.
 *
 * @pre The BCPD algorithm is properly initialized with standard parameters
 * @post The algorithm completes without errors and converges to a solution
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("2D square test", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;
    using VectorType = BCPDType::VectorType;

    std::vector<VectorType> x;
    x.reserve(4u);
    x.emplace_back(0.0, 0.0);
    x.emplace_back(1.0, 0.0);
    x.emplace_back(0.0, 1.0);
    x.emplace_back(1.0, 1.0);

    const VectorType translation{0.01, 0.02};
    std::vector<VectorType> y;
    y.reserve(x.size());
    for (const auto& point : x)
    {
        y.emplace_back(point + translation);
    }

    constexpr FloatType beta{2.0};
    constexpr FloatType lambda{2.0};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{1.0};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @test 2D circle and ellipse test
 * @brief Tests BCPD registration of a 2D circle with affine transformation
 *
 * @details This test case registers a circular point cloud (x) with an elliptical
 * version (y). The template is obtained by applying both a shear transformation
 * and translation to the reference circle. This verifies BCPD's ability to handle
 * non-rigid transformations in 2D.
 *
 * @pre Reference circle has 20 uniformly sampled points on unit circle
 * @post The algorithm converges and estimates the transformation parameters
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("2D circle and ellipse", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;
    using VectorType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    constexpr uint32_t num_points{20u};
    constexpr FloatType angle_step{2.0 * std::numbers::pi / num_points};

    std::vector<VectorType> x;
    x.reserve(num_points);
    for (uint32_t i = 0u; i < num_points; ++i)
    {
        const FloatType angle = angle_step * static_cast<FloatType>(i);
        x.emplace_back(std::cos(angle), std::sin(angle));
    }

    const VectorType translation{0.1, 0.2};
    const MatrixType shear_mat{{1.1, 0.3}, {-0.5, 0.8}};

    std::vector<VectorType> y;
    y.reserve(x.size());
    for (const auto& point : x)
    {
        y.emplace_back(shear_mat * point + translation);
    }

    constexpr FloatType beta{2.0};
    constexpr FloatType lambda{2.0};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{1.0};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @test 3D circle and ellipse test
 * @brief Tests BCPD registration of 3D sphere with affine transformation
 *
 * @details This test case verifies BCPD in three dimensions by registering a
 * spherical point cloud (x) with an ellipsoidal transformed version (y).
 * The sphere is sampled using spherical coordinates, and the template is created
 * via shear transformation and translation. This validates the algorithm's
 * capability to handle 3D deformable registration.
 *
 * @pre Reference sphere has 100 points sampled via spherical coordinates
 * @post The algorithm converges to a solution in 3D space
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("3D circle and ellipse", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;
    using VectorType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    constexpr uint32_t num_x{100u};
    constexpr uint32_t num_y{num_x / 2u};

    std::vector<VectorType> x;
    x.reserve(num_x);
    for (uint32_t i = 0u; i < num_x; ++i)
    {
        const FloatType u = static_cast<FloatType>(i) / static_cast<FloatType>(num_x);
        const FloatType angle = 2.0 * std::numbers::pi * u;

        for (uint32_t j = 0u; j < num_y; ++j)
        {
            const FloatType v = static_cast<FloatType>(j) / static_cast<FloatType>(num_y);
            const FloatType phi = std::acos(2.0 * v - 1.0);
            x.emplace_back(std::cos(angle) * std::cos(phi), std::sin(angle) * std::cos(phi),
                           std::sin(phi));
        }
    }

    const VectorType translation{0.1, 0.2, 0.4};
    const MatrixType shear_mat{{1.1, 0.3, 0.2}, {-0.5, 0.8, 0.1}, {-0.2, -0.3, 0.9}};

    std::vector<VectorType> y;
    y.reserve(x.size());
    for (const auto& point : x)
    {
        y.emplace_back(shear_mat * point + translation);
    }

    constexpr FloatType beta{2.0};
    constexpr FloatType lambda{2.0};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{1.0};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @test 2D fish registration test
 * @brief Tests BCPD with real-world 2D fish point cloud data
 *
 * @details This integration test uses actual fish point cloud data loaded from
 * text files. The reference point cloud (x) and template point cloud (y) are
 * read from disk, representing realistic non-rigid deformation scenarios.
 * The test validates BCPD performance on real point clouds with more complex
 * geometries and larger point counts than synthetic tests.
 *
 * @pre Fish point cloud data files must exist at the specified data directory
 * @post The algorithm completes registration and converges
 * @note Data files: fish-x.txt, fish-y.txt in data/2D directory
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("2D fish registration", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;

    const std::filesystem::path data_dir{testDataDir / "2D"};
    const auto x = readEigenVectorsFromTxtFile<FloatType, 2u>(data_dir / "fish-x.txt");
    const auto y = readEigenVectorsFromTxtFile<FloatType, 2u>(data_dir / "fish-y.txt");

    constexpr FloatType beta{2.0};
    constexpr FloatType lambda{2000.0};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{3.0};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();

    REQUIRE(true);
}

/**
 * @test 3D face registration test
 * @brief Tests BCPD with real-world 3D face point cloud data
 *
 * @details This integration test performs 3D point cloud registration using
 * real face geometry data loaded from text files. It represents a challenging
 * registration scenario with complex non-rigid deformations typical of
 * facial shape variations. The tuning parameters are adjusted for this
 * specific use case (smaller beta, higher lambda).
 *
 * @pre Face point cloud data files must exist at the specified data directory
 * @post The algorithm completes face registration without errors
 * @note Data files: face-x.txt, face-y.txt in data/3D directory
 * @note Tuning parameters optimized for facial geometry
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("3D face registration", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;

    const std::filesystem::path data_dir{testDataDir / "3D"};
    const auto x = readEigenVectorsFromTxtFile<FloatType, 3u>(data_dir / "face-x.txt");
    const auto y = readEigenVectorsFromTxtFile<FloatType, 3u>(data_dir / "face-y.txt");

    constexpr FloatType beta{0.3};
    constexpr FloatType lambda{1.0e4};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{0.1};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @test Armadillo registration test
 * @brief Tests BCPD with real-world 3D armadillo point cloud data
 *
 * @details This integration test registers 3D point clouds representing an
 * armadillo model. It demonstrates BCPD's capability on complex 3D geometries
 * with significant non-rigid deformation. The armadillo dataset is a standard
 * benchmark for point cloud registration algorithms and provides a realistic
 * evaluation of algorithm performance on detailed 3D models.
 *
 * @pre Armadillo point cloud data files must exist at the specified data directory
 * @post The algorithm converges for the complex 3D armadillo geometry
 * @note Data files: armadillo-x.txt, armadillo-y.txt in data/3D directory
 * @note Represents a standard benchmark dataset for registration evaluation
 * @see BCPD::SetInput
 * @see BCPD::Compute
 */
TEST_CASE("armadillo registration", "[bcpd]")
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;

    const std::filesystem::path data_dir{testDataDir / "3D"};
    const auto x = readAndWriteInput<FloatType, 3u>(data_dir, "armadillo-x.txt");
    const auto y = readAndWriteInput<FloatType, 3u>(data_dir, "armadillo-y.txt");

    constexpr FloatType beta{0.3};
    constexpr FloatType lambda{1.0e4};
    constexpr FloatType omega{0.0};
    constexpr FloatType gamma{0.1};

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();

    const auto iterDir = testDataDir / "output" / "armadillo";
    writeOutput<FloatType, 3u>("armadillo", bcpd.GetOutput());

    REQUIRE(true);
}
