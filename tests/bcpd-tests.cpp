/**
 * @file bcpd-tests.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "bcpd.h"
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <numbers>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>


// Utilities
namespace
{
    /**
     * @brief 
     * 
     * @param filename 
     * @return std::vector<VectorXd> 
     */
    template <class FloatType,  uint32_t dim>
    std::vector<Eigen::Vector<FloatType, dim>> readEigenVectorsFromTxtFile(const std::filesystem::path& filename) 
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            exit(1);
        }

        std::vector<Eigen::Vector<FloatType, dim>> vectors;
        std::string line;
        while (std::getline(file, line)) 
        {
            std::stringstream ss(line);
            Eigen::Vector<FloatType, dim> vectorXD;
            FloatType value;
            uint32_t i = 0u;
            while (ss >> value) 
            {
                vectorXD[i] = value;
                i++;
            }
            vectors.push_back(vectorXD);
        }

        file.close();

        return vectors;
    }
}

/**
 * @brief 
 * 
 */
TEST_CASE("2D squre test", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;
    using vType = BCPDType::VectorType;

    std::mt19937 gen(123); // Standard mersenne_twister_engine.
    std::uniform_real_distribution<> dis(-.1, 0.1);

    std::vector<vType> x(4);
    x[0] = vType(0.0, 0.0);
    x[1] = vType(1.0, 0.0);
    x[2] = vType(0.0, 1.0);
    x[3] = vType(1.0, 1.0);

    vType translation(0.01, 0.02);
    std::vector<vType> y(4);
    for(uint32_t i = 0u; i < 4; i++)
    {
        y[i] = x[i] + (1.0 - dis(gen)) * translation;
    }
    
    /*y[0] = vType(0.0, 0.0) + tanslation;
    y[1] = vType(2.0, 0.0) + tanslation;
    y[2] = vType(0.0, 1.0) + tanslation;
    y[3] = vType(2.0, 1.0) + tanslation;*/

    FloatType beta(2.0);
    FloatType lambda(2.0);
    FloatType omega(0.0);
    FloatType gamma(1.0);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @brief 
 * 
 */
TEST_CASE("2D circle and elipse", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;
    using vType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    uint32_t numX = 20u;

    std::mt19937 gen(123); // Standard mersenne_twister_engine.
    std::uniform_real_distribution<> dis(0.0, 2.0 * std::numbers::pi);

    std::vector<vType> x(numX);
    for(uint32_t i = 0; i < numX; i++)
    {
        double angle = 2.0 * std::numbers::pi / double(numX) * double(i);
        x[i] = vType(std::cos(angle), std::sin(angle));
    }

    vType tanslation(0.1, 0.2);
    MatrixType shearMat{{1.1, 0.3}, {-0.5, 0.8}};
    
    std::vector<vType> y(numX);
    for(uint32_t i = 0u; i < numX; i++)
    {
        y[i] = shearMat * x[i] + tanslation;
    }

    FloatType beta(2.0);
    FloatType lambda(2.0);
    FloatType omega(0.0);
    FloatType gamma(1.0);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @brief 
 * 
 */
TEST_CASE("3D circle and elipse", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;
    using vType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    uint32_t numX = 100u;

    std::mt19937 gen(123); // Standard mersenne_twister_engine.
    std::uniform_real_distribution<> dis(0.0, 2.0 * std::numbers::pi);

    std::vector<vType> x(numX);
    for(uint32_t i = 0; i < numX; i++)
    {
        double u =  double(i) / double(numX);
        double angle = 2.0 * std::numbers::pi * u;

        for(uint32_t j = 0u; j < numX / 2u; j++)
        {
            double v = double(j) / double(numX / 2u);
            double phi = std::acos(2 * v - 1);
            x[i] = vType(std::cos(angle) * std::cos(phi), 
                        std::sin(angle) * std::cos(phi),
                        std::sin(phi));
        }
    }

    vType tanslation(0.1, 0.2, 0.4);
    MatrixType shearMat{{1.1, 0.3, 0.2}, 
                        {-0.5, 0.8, 0.1},
                        {-0.2, -0.3, 0.9}};
    
    std::vector<vType> y(numX);
    for(uint32_t i = 0u; i < numX; i++)
    {
        y[i] = shearMat * x[i] + tanslation;
    }

    FloatType beta(2.0);
    FloatType lambda(2.0);
    FloatType omega(0.0);
    FloatType gamma(1.0);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @brief 
 * 
 */
TEST_CASE("2D fish registration", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 2u>;
    using vType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    std::filesystem::path xPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/2D/fish-x.txt");
    std::filesystem::path yPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/2D/fish-y.txt");
    std::vector<vType> x = readEigenVectorsFromTxtFile<FloatType, 2u>(xPath);
    std::vector<vType> y = readEigenVectorsFromTxtFile<FloatType, 2u>(yPath);

    FloatType beta(2.0);
    FloatType lambda(2000.0);
    FloatType omega(0.0);
    FloatType gamma(3.0);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

/**
 * @brief 
 * 
 */
TEST_CASE("3D face registration", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;
    using vType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    std::filesystem::path xPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/3D/face-x.txt");
    std::filesystem::path yPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/3D/face-y.txt");
    std::vector<vType> x = readEigenVectorsFromTxtFile<FloatType, 3u>(xPath);
    std::vector<vType> y = readEigenVectorsFromTxtFile<FloatType, 3u>(yPath);

    FloatType beta(0.3);
    FloatType lambda(1.0e4);
    FloatType omega(0.0);
    FloatType gamma(0.1);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}

TEST_CASE("armadillo_registration", "[bcpd]" ) 
{
    using FloatType = double;
    using BCPDType = BCPD<FloatType, 3u>;
    using vType = BCPDType::VectorType;
    using MatrixType = BCPDType::MatrixType;

    std::filesystem::path xPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/3D/armadillo-x.txt");
    std::filesystem::path yPath("/home/hulfeldl/Documents/10_Code/BCPD/tests/data/3D/armadillo-y.txt");
    std::vector<vType> x = readEigenVectorsFromTxtFile<FloatType, 3u>(xPath);
    std::vector<vType> y = readEigenVectorsFromTxtFile<FloatType, 3u>(yPath);

    FloatType beta(0.3);
    FloatType lambda(1.0e4);
    FloatType omega(0.0);
    FloatType gamma(0.1);

    BCPDType bcpd(beta, lambda, omega, gamma);
    bcpd.SetInput(x, y);
    bcpd.Compute();
    REQUIRE(true);
}
