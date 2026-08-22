/**
 * @file BCPD.cpp
 * @brief Bayesian Coherent Point Drift (BCPD) point cloud registration implementation.
 *
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 */

#include "BCPD.h"

#include <bcpd/GaussianKernel.h>
#include <nanoflann.hpp>
#include <spdlog/spdlog.h>
#include <tinyply.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <source_location>
#include <sstream>
#include <vector>

namespace
{
    constexpr bool kNystrom{true};
    constexpr bool kVisualize{false};
    constexpr bool kWriteDebugOutput{true};
    constexpr bool kPrintDebug{false};
}  // namespace

namespace
{

    std::filesystem::path getDebugOutputDir()
    {
        if (const char* envDir = std::getenv("BCPD_DEBUG_DIR"))
        {
            return envDir;
        }

        return "./debug_output";
    }

    // Hardcoded constants.
    const std::filesystem::path OUTPUT_DIR = getDebugOutputDir();
    constexpr float RESIDUAL_CONVERGENCE_THRESHOLD = 0.001f;
    constexpr uint32_t MAX_ITERATIONS = 100u;
    constexpr float RESIDUAL_KDTREE_THRESHOLD = 0.04f;
    constexpr float SEARCH_RADIUS_SCALE = 5.0f;
    constexpr float SEARCH_RADIUS_MAX = 0.1f;
    constexpr uint32_t KD_TREE_MAX_LEAF = 10u;
    constexpr uint32_t NEAREST_NEIGHBORS_FALLBACK = 5u;
    constexpr float EPSILON_REGULARIZATION = 1.0e-10f;

    // fmt (used by spdlog) cannot auto-format Eigen's expression-template types
    // (e.g. Transpose<...>), so materialize to a string via operator<< first.
    template <typename Derived>
    [[nodiscard]] std::string eigenToString(const Eigen::EigenBase<Derived>& mat)
    {
        std::ostringstream oss;
        oss << mat.derived();
        return oss.str();
    }

    class TimeTracker
    {
    public:
        explicit TimeTracker(std::string_view trackerName) noexcept
            : name(trackerName), start(std::chrono::high_resolution_clock::now())
        {
        }

        ~TimeTracker()
        {
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            spdlog::debug("{} took: {}ms", name, duration.count());
        }

    private:
        std::string name;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
    };

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
        std::string filename = path.string() + "-binary.ply";
        fbBinary.open(filename, std::ios::out | std::ios::binary);
        std::ostream outstream(&fbBinary);

        if (outstream.fail())
        {
            throw std::runtime_error("Failed to open " + filename);
        }
        file.write(outstream, true);
    }

    template <typename FloatType> [[nodiscard]] FloatType digamma(FloatType x_in) noexcept
    {
        double x = static_cast<double>(x_in);
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

        return static_cast<FloatType>(r + std::log(x) - 0.5 / x + t);
    }

    template <typename FloatType>
    [[nodiscard]] FloatType calculateP(FloatType xyDist2, FloatType residual2, FloatType alpha,
                                       FloatType omega) noexcept
    {
        constexpr FloatType TWO_PI = 2.0 * std::numbers::pi_v<FloatType>;
        const FloatType phi_mn =
            (1.0 / std::sqrt(TWO_PI * residual2)) * std::exp(-xyDist2 / (2.0 * residual2));
        return (1.0 - omega) * alpha * phi_mn;
    }

    template <typename FloatType, uint32_t Dim>
    void calculateNystromApprox(
        const Kernel<FloatType, Dim>& kernel, const std::vector<Eigen::Vector<FloatType, Dim>>& y,
        uint32_t kSamples, Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>& eigenVectors,
        Eigen::DiagonalMatrix<FloatType, Eigen::Dynamic>& eigenValues)
    {

        TimeTracker tr("calculateNystromApprox");

        std::vector<uint32_t> indices(y.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        using MatrixType = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;

        MatrixType kernelMat(kSamples, kSamples);
        for (uint32_t i = 0u; i < kSamples; ++i)
        {
            for (uint32_t j = i; j < kSamples; ++j)
            {
                const FloatType kernelVal = kernel.compute(y[indices[i]], y[indices[j]]);
                kernelMat(i, j) = kernelMat(j, i) = kernelVal;
            }
        }

        Eigen::JacobiSVD<MatrixType> svd(kernelMat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        spdlog::debug("Singular values: {}", eigenToString(svd.singularValues().transpose()));

        const auto U = svd.matrixU();
        eigenValues.resize(kSamples);
        eigenValues = svd.singularValues().asDiagonal();
        eigenValues.diagonal().array() += EPSILON_REGULARIZATION;

        MatrixType kernelMxK(y.size(), kSamples);
        for (uint32_t m = 0u; m < y.size(); ++m)
        {
            for (uint32_t k = 0u; k < kSamples; ++k)
            {
                kernelMxK(m, k) = kernel.compute(y[m], y[indices[k]]);
            }
        }

        eigenVectors = kernelMxK * U * eigenValues.inverse();
    }

}  // namespace

template <typename FloatType, uint32_t Dim> inline void BCPD<FloatType, Dim>::Compute()
{
    auto normalizePoints = [this](std::vector<VectorType>& points) -> void
    {
        if (points.empty())
            return;

        VectorType centroid = VectorType::Zero();
        for (const auto& p : points)
        {
            centroid += p;
        }
        centroid /= static_cast<FloatType>(points.size());

        FloatType scale = 0.0;
        for (const auto& p : points)
        {
            scale += (p - centroid).squaredNorm();
        }
        scale = std::sqrt(scale / (static_cast<FloatType>(points.size() * Dim)));

        for (auto& p : points)
        {
            p = (p - centroid) / scale;
        }
    };

    normalizePoints(x);
    normalizePoints(y);

    Initialization();

    iter = 0u;
    spdlog::info("Iteration: {} residual: {}", iter, residual);

    while (residual > RESIDUAL_CONVERGENCE_THRESHOLD && iter < MAX_ITERATIONS)
    {
        TimeTracker tr("Iteration");
        ExpectationStep();
        MaximizationStep();
        ++iter;
        spdlog::info("Iteration: {} residual: {}", iter, residual);
    }
}

template <typename FloatType, uint32_t Dim> inline void BCPD<FloatType, Dim>::Initialization()
{
    const uint32_t N = x.size();
    const uint32_t M = y.size();

    TimeTracker tr("Initialization");

    m_kernel = std::make_unique<GaussKernel<FloatType, Dim>>(m_beta);

    alpha.assign(M, 1.0 / static_cast<FloatType>(M));

    scale = 1.0;
    rotation.setIdentity();
    translation.setZero();

    y_hat = y;

    residual = FloatType(0.0);
    for (uint32_t m = 0u; m < M; ++m)
    {
        residual += N * y[m].squaredNorm();
    }

    for (uint32_t n = 0u; n < N; ++n)
    {
        residual += M * x[n].squaredNorm();
    }

    for (uint32_t d = 0u; d < Dim; ++d)
    {
        FloatType sumY = 0.0;
        FloatType sumX = 0.0;

        for (uint32_t m = 0u; m < M; ++m)
        {
            sumY += y[m][d];
        }
        for (uint32_t n = 0u; n < N; ++n)
        {
            sumX += x[n][d];
        }

        residual -= 2.0 * sumY * sumX;
    }

    residual *= (m_gamma * m_gamma) / static_cast<FloatType>(M * N * Dim);

    if constexpr (!kNystrom)
    {
        sigma.setIdentity(M, M);
        G.resize(M, M);
        for (uint32_t i = 0u; i < M; ++i)
        {
            for (uint32_t j = i; j < M; ++j)
            {
                const FloatType kernelVal = m_kernel->compute(y[i], y[j]);
                G(i, j) = G(j, i) = kernelVal;
            }
        }
    }
    else
    {
        xKdTree = std::make_unique<kdTreeType>(Dim, x, KD_TREE_MAX_LEAF);
        auto numSamples = std::min(kSamples, static_cast<uint32_t>(y.size()));
        calculateNystromApprox<FloatType, Dim>(*m_kernel, y, numSamples, Q, LAMBDA);
    }
}

template <typename FloatType, uint32_t Dim> inline void BCPD<FloatType, Dim>::ExpectationStep()
{
    TimeTracker tr("ExpectationStep");

    const uint32_t N = x.size();
    const uint32_t M = y.size();
    sigmaSQR = residual;

    if constexpr (kNystrom)
    {
        if (residual > RESIDUAL_KDTREE_THRESHOLD)
        {
            computeExpectationNystrom(N, M);
        }
        else
        {
            computeExpectationKdTree(N, M);
        }
    }
    else
    {
        computeExpectationDirect(N, M);
    }

    for (uint32_t m = 0u; m < M; ++m)
    {
        alpha[m] = std::exp(digamma(m_kappa + nu[m]) -
                            digamma(m_kappa * static_cast<FloatType>(M) + N_hat));
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::computeExpectationNystrom(uint32_t N, uint32_t M)
{
    const uint32_t numVSamples = std::min(2u * (vSamples / 2u), static_cast<uint32_t>(x.size()));

    std::vector<uint32_t> indicesX(N);
    std::vector<uint32_t> indicesY(M);
    std::iota(indicesX.begin(), indicesX.end(), 0u);
    std::iota(indicesY.begin(), indicesY.end(), 0u);

    std::shuffle(indicesX.begin(), indicesX.end(), std::mt19937{std::random_device{}()});
    std::shuffle(indicesY.begin(), indicesY.end(), std::mt19937{std::random_device{}()});

    spdlog::debug("Using Nystrom method for P computation");

    std::vector<VectorType> v(numVSamples);
    const uint32_t sampleCount = numVSamples / 2u;
    for (uint32_t i = 0u; i < sampleCount; ++i)
    {
        v[2u * i] = x[indicesX[i]];
        v[2u * i + 1u] = y_hat[indicesY[i]];
    }

    EigenMatrix kernelVxV(numVSamples, numVSamples);
    for (uint32_t i = 0u; i < numVSamples; ++i)
    {
        for (uint32_t j = i; j < numVSamples; ++j)
        {
            const FloatType kernelVal = std::exp(-(v[i] - v[j]).squaredNorm() / (2.0 * residual));
            kernelVxV(i, j) = kernelVxV(j, i) = kernelVal;
        }
    }

    EigenMatrix kernelXxV(N, numVSamples);
    for (uint32_t n = 0u; n < N; ++n)
    {
        for (uint32_t k = 0u; k < numVSamples; ++k)
        {
            kernelXxV(n, k) = std::exp(-(x[n] - v[k]).squaredNorm() / (2.0 * residual));
        }
    }

    const EigenMatrix kernelVxX = kernelVxV.inverse() * kernelXxV.transpose();

    EigenMatrix kernelYxV(M, numVSamples);
    for (uint32_t m = 0u; m < M; ++m)
    {
        for (uint32_t k = 0u; k < numVSamples; ++k)
        {
            kernelYxV(m, k) = std::exp(-(y_hat[m] - v[k]).squaredNorm() / (2.0 * residual));
        }
    }

    std::vector<FloatType> b(M);
    for (uint32_t m = 0u; m < M; ++m)
    {
        b[m] = alpha[m] * std::exp(-std::pow(scale, 2.0) / (2.0 * residual) * FloatType(Dim));
        assert(b[m] != FloatType(0.0));
    }

    std::vector<FloatType> qV(numVSamples, FloatType(0.0));
    for (uint32_t k = 0u; k < numVSamples; ++k)
    {
        for (uint32_t m = 0u; m < M; ++m)
        {
            qV[k] += kernelYxV(m, k) * b[m];
        }
    }

    std::vector<FloatType> q(N, FloatType(0.0));
    for (uint32_t n = 0u; n < N; ++n)
    {
        for (uint32_t k = 0u; k < numVSamples; ++k)
        {
            q[n] += kernelVxX(k, n) * qV[k];
        }
        q[n] = 1.0 / q[n];
    }

    nu.assign(M, FloatType(0.0));
    N_hat = 0.0;

    std::vector<FloatType> nuV(numVSamples, FloatType(0.0));
    for (uint32_t k = 0u; k < numVSamples; ++k)
    {
        for (uint32_t n = 0u; n < N; ++n)
        {
            nuV[k] += kernelVxX(k, n) * q[n];
        }
    }

    for (uint32_t m = 0u; m < M; ++m)
    {
        for (uint32_t k = 0u; k < numVSamples; ++k)
        {
            nu[m] += kernelYxV(m, k) * nuV[k];
        }
        nu[m] *= b[m];
        N_hat += nu[m];
    }

    nu_apo.assign(N, FloatType(1.0));

    std::vector<VectorType> PxV(numVSamples, VectorType::Zero());
    for (uint32_t k = 0u; k < numVSamples; ++k)
    {
        for (uint32_t n = 0u; n < N; ++n)
        {
            PxV[k] += kernelVxX(k, n) * q[n] * x[n];
        }
    }

    Px.assign(M, VectorType::Zero());
    for (uint32_t m = 0u; m < M; ++m)
    {
        for (uint32_t k = 0u; k < numVSamples; ++k)
        {
            Px[m] += kernelYxV(m, k) * PxV[k];
        }
        Px[m] *= b[m];
    }

    x_hat.resize(M);
    for (uint32_t m = 0u; m < M; ++m)
    {
        x_hat[m] = Px[m] / nu[m];
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::computeExpectationKdTree(uint32_t N, uint32_t M)
{
    spdlog::debug("Using kd-tree method for P computation");

    const FloatType searchRadius =
        std::min(static_cast<FloatType>(SEARCH_RADIUS_SCALE) * std::sqrt(residual),
                 static_cast<FloatType>(SEARCH_RADIUS_MAX));

    std::vector<std::vector<std::pair<std::size_t, FloatType>>> P_kd(M);

    for (uint32_t m = 0u; m < M; ++m)
    {
        std::vector<nanoflann::ResultItem<std::size_t, FloatType>> ret_matches;
        xKdTree->radiusSearch(y_hat[m].data(), searchRadius, ret_matches);

        if (ret_matches.empty())
        {
            std::vector<std::size_t> ret_index(NEAREST_NEIGHBORS_FALLBACK);
            std::vector<FloatType> out_dist_sqr(NEAREST_NEIGHBORS_FALLBACK);

            xKdTree->query(y_hat[m].data(), NEAREST_NEIGHBORS_FALLBACK, ret_index.data(),
                           out_dist_sqr.data());

            for (uint32_t i = 0u; i < NEAREST_NEIGHBORS_FALLBACK; ++i)
            {
                ret_matches.emplace_back(ret_index[i], out_dist_sqr[i]);
            }
            spdlog::warn("No matches found at m={}", m);
        }

        for (const auto& xNeighbor : ret_matches)
        {
            const FloatType p_mn = calculateP(xNeighbor.second, residual, alpha[m], m_omega);
            P_kd[m].emplace_back(xNeighbor.first, p_mn);
        }
    }

    std::vector<FloatType> sumP(N, FloatType(0.0));
    for (uint32_t m = 0u; m < M; ++m)
    {
        for (const auto& [idx, pVal] : P_kd[m])
        {
            sumP[idx] += pVal;
        }
    }

    for (uint32_t m = 0u; m < M; ++m)
    {
        for (auto& [idx, pVal] : P_kd[m])
        {
            pVal /= sumP[idx];
        }
    }

    nu.assign(M, FloatType(0.0));
    nu_apo.assign(N, FloatType(0.0));
    N_hat = 0.0;

    for (uint32_t m = 0u; m < M; ++m)
    {
        FloatType sumPm = 0.0;
        for (const auto& [idx, pVal] : P_kd[m])
        {
            nu_apo[idx] += pVal;
            sumPm += pVal;
        }
        nu[m] = sumPm;
        N_hat += sumPm;
    }

    Px.assign(M, VectorType::Zero());
    x_hat.resize(M);

    for (uint32_t m = 0u; m < M; ++m)
    {
        for (const auto& [idx, pVal] : P_kd[m])
        {
            Px[m] += pVal * x[idx];
        }

        if (nu[m] != FloatType(0.0))
        {
            x_hat[m] = Px[m] / nu[m];
        }
        else
        {
            spdlog::warn("nu[{}] is zero", m);
            x_hat[m] = Px[m];
        }
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::computeExpectationDirect(uint32_t N, uint32_t M)
{
    P.resize(N);
    for (uint32_t n = 0u; n < N; ++n)
    {
        P[n].resize(M);
    }

    for (uint32_t n = 0u; n < N; ++n)
    {
        FloatType sum_p_mn = 0.0;

        for (uint32_t m = 0u; m < M; ++m)
        {
            const FloatType xyDist2 = (x[n] - y_hat[m]).squaredNorm();
            const FloatType p_mn = calculateP(xyDist2, residual, alpha[m], m_omega);
            P[n][m] = p_mn;
            sum_p_mn += p_mn;
        }

        for (uint32_t m = 0u; m < M; ++m)
        {
            P[n][m] /= (m_omega * p_out + sum_p_mn);
        }
    }

    nu.assign(M, FloatType(0.0));
    nu_apo.assign(N, FloatType(0.0));
    N_hat = 0.0;

    for (uint32_t m = 0u; m < M; ++m)
    {
        FloatType sumPn = 0.0;
        for (uint32_t n = 0u; n < N; ++n)
        {
            sumPn += P[n][m];
            nu_apo[n] += P[n][m];
        }
        nu[m] = sumPn;
        N_hat += sumPn;
    }

    x_hat.resize(M);
    for (uint32_t m = 0u; m < M; ++m)
    {
        VectorType x_hat_m = VectorType::Zero();
        for (uint32_t n = 0u; n < N; ++n)
        {
            x_hat_m += P[n][m] * x[n];
        }
        x_hat[m] = x_hat_m / nu[m];
    }
}

template <typename FloatType, uint32_t Dim> inline void BCPD<FloatType, Dim>::MaximizationStep()
{
    TimeTracker tr("MaximizationStep");

    const uint32_t N = x.size();
    const uint32_t M = y.size();

    std::vector<VectorType> E(M);
    for (uint32_t i = 0u; i < M; ++i)
    {
        E[i] = (rotation.transpose() * (x_hat[i] - translation)) / scale - y[i];
    }

    const FloatType cc = (scale * scale) / residual;

    if constexpr (kNystrom)
    {
        computeMaximizationNystrom(M, cc, E);
    }
    else
    {
        computeMaximizationDirect(M, cc, E);
    }

    for (uint32_t i = 0u; i < M; ++i)
    {
        y_hat[i] = scale * rotation * u_hat[i] + translation;
    }

    updateResidual(N, M);

    if constexpr (kWriteDebugOutput)
    {
        writeDebugOutput();
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::computeMaximizationNystrom(uint32_t M, FloatType cc,
                                                      const std::vector<VectorType>& E)
{
    Eigen::Vector<FloatType, Eigen::Dynamic> nuVec(M);
    for (uint32_t j = 0u; j < M; ++j)
    {
        nuVec[j] = nu[j];
    }

    const EigenMatrix S = Q.transpose() * nuVec.asDiagonal() * Q;
    EigenMatrix sigmaKxK = S + (m_lambda / cc) * EigenMatrix(LAMBDA.inverse());

    const EigenMatrix sigmaKxM = sigmaKxK.inverse() * Q.transpose();

    const auto numSamples = std::min(kSamples, static_cast<uint32_t>(E.size()));
    std::vector<VectorType> v_hatK(numSamples, VectorType::Zero());
    for (uint32_t i = 0u; i < numSamples; ++i)
    {
        for (uint32_t j = 0u; j < M; ++j)
        {
            v_hatK[i] += cc * sigmaKxM(i, j) * nu[j] * E[j];
        }
    }

    u_hat.resize(M);
    for (uint32_t i = 0u; i < M; ++i)
    {
        VectorType v_hat_i = VectorType::Zero();
        for (uint32_t j = 0u; j < numSamples; ++j)
        {
            v_hat_i += Q(i, j) * v_hatK[j];
        }
        u_hat[i] = y[i] + v_hat_i;
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::computeMaximizationDirect(uint32_t M, FloatType cc,
                                                     const std::vector<VectorType>& E)
{
    sigma.setZero(M, M);
    for (uint32_t i = 0u; i < M; ++i)
    {
        for (uint32_t j = i; j < M; ++j)
        {
            sigma(i, j) = (G(i, j) / m_lambda) * cc * cc * nu[i] * nu[j];
            if (i == j)
            {
                sigma(i, j) += cc * nu[i];
            }
            else
            {
                sigma(j, i) = sigma(i, j);
            }
        }
    }

    sigma = sigma.inverse();

    for (uint32_t i = 0u; i < M; ++i)
    {
        for (uint32_t j = i; j < M; ++j)
        {
            sigma(i, j) = -sigma(i, j);
            if (i == j)
            {
                sigma(i, j) += 1.0 / (cc * nu[i]);
            }
            else
            {
                sigma(j, i) = sigma(i, j);
            }
        }
    }

    spdlog::debug("sigma:\n{}", eigenToString(sigma));

    u_hat.resize(M);
    for (uint32_t i = 0u; i < M; ++i)
    {
        VectorType v_hat = VectorType::Zero();
        for (uint32_t j = 0u; j < M; ++j)
        {
            v_hat += cc * sigma(i, j) * nu[j] * E[j];
        }
        u_hat[i] = y[i] + v_hat;
    }
}

template <typename FloatType, uint32_t Dim>
void BCPD<FloatType, Dim>::updateResidual(uint32_t N, uint32_t M)
{
    spdlog::debug("u_hat computed");

    VectorType x_avg = VectorType::Zero();
    for (uint32_t i = 0u; i < M; ++i)
    {
        x_avg += nu[i] * x_hat[i];
    }
    x_avg /= N_hat;

    VectorType u_avg = VectorType::Zero();
    for (uint32_t i = 0u; i < M; ++i)
    {
        u_avg += nu[i] * u_hat[i];
    }
    u_avg /= N_hat;

    MatrixType Sxu = MatrixType::Zero();
    for (uint32_t i = 0u; i < M; ++i)
    {
        Sxu += nu[i] * (x_hat[i] - x_avg) * (u_hat[i] - u_avg).transpose();
    }
    Sxu /= N_hat;

    MatrixType Suu = MatrixType::Zero();
    for (uint32_t i = 0u; i < M; ++i)
    {
        Suu += nu[i] * (u_hat[i] - u_avg) * (u_hat[i] - u_avg).transpose();
    }
    Suu /= N_hat;

    Eigen::JacobiSVD<MatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(
        Sxu, Eigen::ComputeFullU | Eigen::ComputeFullV);

    spdlog::debug("Sxu singular values: {}", eigenToString(svd.singularValues().transpose()));

    const auto U = svd.matrixU();
    const auto V = svd.matrixV();

    MatrixType R_diag = MatrixType::Identity(Dim, Dim);
    R_diag(Dim - 1u, Dim - 1u) = (U * V.transpose()).determinant();
    rotation = U * R_diag * V.transpose();

    spdlog::debug("Rotation determinant: {}", rotation.determinant());

    scale = (rotation.transpose() * Sxu).trace() / Suu.trace();
    translation = x_avg - scale * rotation * u_avg;

    if constexpr (kNystrom)
    {
        residual = FloatType(0.0);
        for (uint32_t i = 0u; i < N; ++i)
        {
            residual += nu_apo[i] * x[i].squaredNorm();
        }

        for (uint32_t i = 0u; i < M; ++i)
        {
            residual -= 2.0 * Px[i].dot(y_hat[i]);
        }

        for (uint32_t i = 0u; i < M; ++i)
        {
            residual += nu[i] * y_hat[i].squaredNorm();
        }

        residual /= (static_cast<FloatType>(Dim) * N_hat);
    }
    else
    {
        residual = 0.0;
        for (uint32_t i = 0u; i < M; ++i)
        {
            for (uint32_t j = 0u; j < N; ++j)
            {
                residual += P[j][i] * (y_hat[i] - x[j]).squaredNorm();
            }
        }
        residual /= (static_cast<FloatType>(Dim) * N_hat);
    }
}

template <typename FloatType, uint32_t Dim> void BCPD<FloatType, Dim>::writeDebugOutput() const
{
    const std::string folder = "Iteration_" + std::to_string(iter);
    const auto iterDir = OUTPUT_DIR / folder;
    std::filesystem::create_directories(iterDir);

    writePly<FloatType, Dim>(x, iterDir / "x.ply");
    writePly<FloatType, Dim>(x_hat, iterDir / "x_hat.ply");
    writePly<FloatType, Dim>(y, iterDir / "y.ply");
    writePly<FloatType, Dim>(y_hat, iterDir / "y_hat.ply");
}

template class BCPD<double, 2u>;
template class BCPD<double, 3u>;
