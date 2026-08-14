#include <random>
/**
 * @file bcpd.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-08-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "bcpd.h"
#include <complex>
#include <cstdint>
#include <numbers>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <source_location>
#include <cassert>
#include "nanoflann.hpp"
#include "tinyply.h"

#ifndef NDEBUG
    #include <CvPlot/cvplot.h>
#endif

#define NYSTROM 1
#define VISUALIZE 0 // Activate visualization
#define WRITE_DEBUG_OUTPUT 1
#define PRINT_DEBUG 0

/**
 * @brief 
 * 
 */
namespace
{
    // output
    const std::filesystem::path outputDir("/home/hulfeldl/DebugOutput/3D/armadillo/");

    class TimeTracker
    {
        public:

        TimeTracker(const std::string& trackerName)
        :   name(trackerName)
        {
            start = std::chrono::high_resolution_clock::now();

        }

        ~TimeTracker()
        {
            const auto end = std::chrono::high_resolution_clock::now();
 
            const std::chrono::duration<double> diff = end - start;
            const auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            std::cout << name << " took: " << int_ms << "ms \n";
        }

        private:
            std::chrono::time_point<std::chrono::high_resolution_clock> start;
            std::string name;
    };

    template <class FloatType,  uint32_t dim>
    void writePly(const std::vector<Eigen::Vector<FloatType, dim>>& points, std::filesystem::path& path)
    {
        struct double3 { double x, y, z; };
        std::vector<double3> pointsOut;
        for(const auto& point : points)
        {
            pointsOut.push_back({.x = point[0], .y = point[1], .z = point[2]});
        }

        tinyply::PlyFile file;

        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT64, points.size(), reinterpret_cast<uint8_t*>(pointsOut.data()), tinyply::Type::INVALID, 0);

        // Write a binary file
        std::filebuf fb_binary;
        std::string filename = path;
        fb_binary.open(filename + "-binary.ply", std::ios::out | std::ios::binary);
        std::ostream outstream_binary(&fb_binary);
        if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);
        file.write(outstream_binary, true);
    }
}

/**
 * @brief 
 * 
 */
namespace
{
    template <class FloatType>
    FloatType digamma(FloatType x_in)
    {
        double x = double(x_in);
        double r, f, t;

        r = 0;

        while (x<=5)
        { r -= 1/x;
            x += 1;
        }

        f = 1/(x*x);

        t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
            + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))));

        return FloatType(r + log(x) - 0.5/x + t);
    }

    template <class FloatType>
    FloatType dist1(const FloatType x, const FloatType *y, int D)
    {
        int d;
        double val = 0;
        for (d = 0; d < D; d++)
            val += fabs(x[d] - y[d]);
        return val;
    }

    template <class FloatType, class VectorType>
    inline FloatType getDistanceSqr(const VectorType &x, const VectorType &y)
    {
        FloatType lengthSqr = 0.0;
        for (uint32_t i = 0u; i < x.size(); i++)
        {
            for (const auto y_i : y)
            {
                lengthSqr += (x[i] - y[i]) * (x[i] - y[i]);
            }
        }
        return lengthSqr;
    }

    template <class FloatType>
    FloatType calculateP(const FloatType xyDist2, const FloatType residual2, const FloatType sigma_m2,  const FloatType alpha,
    const FloatType omega)
    {
        // calculate phi_mn
        FloatType phi_mn = 1.0 / std::sqrt(2.0 * std::numbers::pi_v<FloatType> * residual2);
        phi_mn *= std::exp( -1.0 / (2.0 * residual2) * xyDist2);
        //phi_mn *= std::exp( -1.0 / (2.0 * residual2) * std::pow(scale, 2.0) * sigma_m2 * FloatType(dim));


        // calculate p_mn
        FloatType p_mn = (1.0 - omega) * alpha * phi_mn;
        return p_mn;
    }

}

/**
 * @brief 
 * 
 * @tparam FloatType 
 */
template <class FloatType, uint32_t dim>
class GaussKernel : public Kernel<FloatType, dim>
{
    public:

        GaussKernel(FloatType beta)
        {
            h = beta;
        }


        FloatType compute(const Kernel<FloatType, dim>::VectorType& x, const Kernel<FloatType, dim>::VectorType& y) const override
        {
            return std::exp( -(x - y).dot(x - y) / (2 * h *h)); 
        }

    private:
        FloatType h = 0.5;
};

template <class FloatType,  uint32_t dim>
void calculateNystromApprox(const Kernel<FloatType,dim>& kernel, 
                            const std::vector<Eigen::Vector<FloatType, dim>>& y,
                            const uint32_t kSamples,
                            Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>& eigenVectors,
                            Eigen::DiagonalMatrix<FloatType,Eigen::Dynamic>& eigenValues
                            )
{
    TimeTracker tr("calculateNystromApprox");

    std::vector<uint32_t> indices(y.size());
    for (int i = 0; i < y.size(); i++) {
      indices[i] = i;
    }

    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});


    using MatrixType = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
    using DiagType = Eigen::DiagonalMatrix<FloatType,Eigen::Dynamic>;
    MatrixType kernelMat(kSamples, kSamples);
    for(uint32_t i = 0u; i < kSamples; i++)
    {
        for(uint32_t j = i; j < kSamples; j++)
        {
            kernelMat(i,j) = kernelMat(j,i) = kernel.compute(y[indices[i]], y[indices[j]]);
        }
    }

    Eigen::JacobiSVD<MatrixType> svd(kernelMat,  Eigen::ComputeFullU | Eigen::ComputeFullV);
    //Eigen::BDCSVD<MatrixType> svd(kernelMat,  Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
    auto U = svd.matrixU();
    //auto V = svd.matrixV();

    eigenValues.resize(kSamples);
    eigenValues = (svd.singularValues()).asDiagonal();
    eigenValues.diagonal().array() += 1.0e-10;

    MatrixType kernelMxK(y.size(), kSamples);
    for(uint32_t m = 0u; m < y.size(); m++)
    {
        for(uint32_t k = 0; k < kSamples; k++)
        {
            kernelMxK(m,k) = kernel.compute(y[m], y[indices[k]]);
        }
    }

    eigenVectors.resize(y.size(), kSamples);
    eigenVectors = kernelMxK * U * eigenValues.inverse();
}

/**
 * @brief 
 * 
 */
template <class FloatType, uint32_t dim>
inline void BCPD<FloatType, dim>::Compute()
{
    // Normalize.
    VectorType x_avg = VectorType::Zero();
    for(const auto& x_i : x)
    {
        x_avg += x_i;
    }
    x_avg /= x.size();

    // Scale
    FloatType x_scale = 0.0;
    for(const auto& x_i : x)
    {
        x_scale += (x_i - x_avg).squaredNorm();
    }
    x_scale /= FloatType(x.size() * dim);
    x_scale = std::sqrt(x_scale);

    for(auto& x_i : x)
    {
        x_i = (x_i - x_avg) / x_scale;
    }

    // Normalize.
    VectorType y_avg = VectorType::Zero();
    for(const auto& y_i : y)
    {
        y_avg += y_i;
    }
    y_avg /= y.size();

    // Scale
    FloatType y_scale = 0.0;
    for(const auto& y_i : y)
    {
        y_scale += (y_i - y_avg).squaredNorm();
    }
    y_scale /= FloatType(y.size() * dim);
    y_scale = std::sqrt(y_scale);

    for(auto& y_i : y)
    {
        y_i = (y_i - y_avg) / y_scale;
    }


    FloatType c_resError = 0.001;

    Initialization();
    
    iter = 0u;
    std::cout << "Iteration: " << iter << " residual is: " << residual << std::endl;
    while(residual > c_resError)
    {
        TimeTracker tr("Iteration");
        ExpectationStep();
        MaximizationStep();
        iter++;

        std::cout << "Iteration: " << iter << " residual is: " << residual << std::endl;
    }


}

/**
 * @brief
 *
 * @tparam FloatType
 */
template <class FloatType, uint32_t dim>
inline void BCPD<FloatType, dim>::Initialization()
{
    uint32_t N =  x.size();
    uint32_t M =  y.size();

    TimeTracker tr("Initialization");

    //Initialize kernel
    m_kernel = std::make_unique<GaussKernel<FloatType, dim>>(m_beta);
    
    /*---------------------------------------------------------------o
    |   initialization                                               |
    o---------------------------------------------------------------*/
    // alpha
    //std::size_t M = y.size();
    alpha.resize(M);
    for (auto &alpha_m : alpha)
    {
        alpha_m = 1.0 / FloatType(M);
    }

    /* s,R,t */
    scale = 1.0;

    for (uint32_t i = 0u; i < dim; i++)
        {
            for (uint32_t j = 0u; j < dim; j++)
            {
                rotation(i,j) = (i == j ? 1 : 0);
            }
        }

    for (auto &elem : translation)
    {
        elem = 0.0;
    }

    /* y, b */
    y_hat = y;

    /*for (uint32_t m = 0; m < M; m++)
    {
        b[m] = 1.0;
    }*/

    /* r */
    residual = FloatType(0.0);

    FloatType val1 = FloatType(0.0);
    FloatType val2 = FloatType(0.0);

#if 0
    for (uint32_t m = 0; m < M; m++)
    {
        for (uint32_t n = 0; n < N; n++)
        {
            residual += (x[n] - y[m]).dot(x[n] - y[m]);
        }
    }
#else
    for (uint32_t m = 0; m < M; m++)
    {
        residual += N * y[m].squaredNorm();
    }
    
    for (uint32_t n = 0; n < N; n++)
    {
        residual += M * x[n].squaredNorm();
    }

    for(uint32_t d = 0u; d < dim; d++)
    {
        FloatType sumY(0.0);
         for (uint32_t m = 0; m < M; m++)
        {
            sumY += y[m][d];
        }
        
        FloatType sumX(0.0);
        for (uint32_t n = 0; n < N; n++)
        {
            sumX += x[n][d];
        }   

        residual -= 2* sumY * sumX;
    }

#endif

    residual *= (m_gamma * m_gamma)  / FloatType(M * N * dim);

    // sigma
#if !NYSTROM
    sigma.resize(M,M);
    sigma.setIdentity();
#endif

#if !NYSTROM
    G.resize(M,M);
    /* G */
    for (uint32_t i = 0; i < M; i++)
    {
        for (uint32_t j = i; j < M; j++)
        {
            G(i, j) = G(j,i) = m_kernel->compute(y[i], y[j]);
        }
    }

#else
    // Create kd-tree.
    xKdTree = std::make_unique<kdTreeType>(dim /*dim*/, x, 10 /* max leaf */);

    calculateNystromApprox<FloatType,dim>(*m_kernel, 
                            y,
                            kSamples, 
                            Q,
                            LAMBDA
                            );
#endif

}

/**
 * @brief
 *
 * @tparam FloatType
 */
template <class FloatType, uint32_t dim>
inline void BCPD<FloatType, dim>::ExpectationStep()
{
    TimeTracker tr("ExpectationStep");

    uint32_t N =  x.size();
    uint32_t M =  y.size();

    sigmaSQR = residual;


    const uint32_t numVSamples = 2u *(vSamples / 2u);
    const FloatType residualKDTree(0.2 * 0.2);
#if NYSTROM

    if(residual > residualKDTree)
    {
        std::cout << "Using nyström method to comput P: " << std::endl;

        // x-indices
        std::vector<uint32_t> indicesX(N);
        for (int i = 0; i < N; i++) {
        indicesX[i] = i;
        }
        std::shuffle(indicesX.begin(), indicesX.end(), std::mt19937{std::random_device{}()});

        // y-indices
        std::vector<uint32_t> indicesY(M);
        for (int i = 0; i < M; i++) {
        indicesY[i] = i;
        }
        std::shuffle(indicesY.begin(), indicesY.end(), std::mt19937{std::random_device{}()});

        std::vector<VectorType> v(numVSamples);
        const uint32_t sampleCount = numVSamples / 2u;
        for (int i = 0; i < sampleCount; i++) 
        {
            v[2*i] = x[indicesX[i]];
            v[2*i+1] = y_hat[indicesY[i]];
        }

        EigenMatrix kernelVxV(numVSamples, numVSamples);
        for(uint32_t i = 0u; i < numVSamples; i++)
        {
            for(uint32_t j = i; j < numVSamples; j++)
            {
                kernelVxV(i,j) = kernelVxV(j,i) = std::exp(-(v[i] - v[j]).squaredNorm() / (2.0 * residual));
            }
        }

        // Kernel XxV
        EigenMatrix kernelXxV(N, numVSamples);
        for(uint32_t n = 0u; n < N; n++)
        {
            for(uint32_t k = 0; k < numVSamples; k++)
            {
                kernelXxV(n,k) = std::exp(-(x[n] - v[k]).squaredNorm() / (2.0 * residual));
            }
        }

        EigenMatrix kernelVxX = kernelVxV.inverse() * kernelXxV.transpose();

        EigenMatrix kernelYxV(M, numVSamples);
        for(uint32_t m = 0u; m < M; m++)
        {
            for(uint32_t k = 0; k < numVSamples; k++)
            {
                kernelYxV(m,k) = std::exp(-(y_hat[m] - v[k]).squaredNorm() / (2.0 * residual));
            }
        }

        // b-vector
        std::vector<FloatType> b(M);
        for(uint32_t m = 0u; m < M; m++)
        {
            
            b[m] = alpha[m] * std::exp(- std::pow(scale, 2.0) / (2.0 * residual) * FloatType(dim) /* sigma(m,m)*/); 
            assert(b[m] != FloatType(0.0));
        }

        // TODO: c-vector is always zero, because omega is currently zero.

        std::vector<FloatType> qV(numVSamples);        
        for(uint32_t k = 0; k < numVSamples; k++)
        {
            for(uint32_t m = 0u; m < M; m++)    
            {
                qV[k] += kernelYxV(m,k) * b[m];
            }
        }

        std::vector<FloatType> q(M);
        for(uint32_t n = 0u; n < N; n++)
        {
            for(uint32_t k = 0; k < numVSamples; k++)  
            {
                q[n] += kernelVxX(k,n) * qV[k];
            }
        }
        for(uint32_t n = 0u; n < N; n++)
        {
            q[n] = 1.0 / q[n];
        }

        // Update nu, N_hat.
        nu.resize(M);
        N_hat = 0.0;  
        std::vector<FloatType> nuV(numVSamples);  
        for(uint32_t k = 0; k < numVSamples; k++)  
        {
            nuV[k] = FloatType(0.0);
            for(uint32_t n = 0u; n < N; n++)
            {
                nuV[k] += kernelVxX(k,n) * q[n];
            }
        }

        for (uint32_t m = 0u; m < M; m++)
        {
            nu[m] = FloatType(0.0);
            for(uint32_t k = 0; k < numVSamples; k++)  
            {
                nu[m] += kernelYxV(m,k) * nuV[k];
            }
            nu[m] *= b[m];
            N_hat += nu[m];
        }

        // Update nu_apo
        nu_apo.assign(N, FloatType(0.0));
        for(uint32_t n = 0u; n < N; n++)
        {
            nu_apo[n] = FloatType(1.0);
        }

        // Compute Px
        std::vector<VectorType> PxV(numVSamples);  
        for(uint32_t k = 0; k < numVSamples; k++)  
        {
            PxV[k] = VectorType::Zero();
            for(uint32_t n = 0u; n < N; n++)
            {
                PxV[k] += kernelVxX(k,n) * q[n] * x[n];
            }
        }

        Px.resize(M);
        for (uint32_t m = 0u; m < M; m++)
        {
            Px[m] = VectorType::Zero();
            for(uint32_t k = 0; k < numVSamples; k++)  
            {
                Px[m] += kernelYxV(m,k) * PxV[k];
            }
            Px[m] *= b[m];
        }

        // Calculate x_hat
        x_hat.resize(M);
        for (uint32_t m = 0u; m < M; m++)
        {
            x_hat[m] = Px[m] / nu[m];
        }

    }
    else
    {
        std::cout << "Using kd-tree method to comput P: " << std::endl;

        FloatType search_radius = FloatType(5.0) * std::sqrt(residual);
        search_radius = std::min(search_radius, 0.1);

        // Calculate P using kdTree.
        std::vector<std::vector<std::pair<std::size_t,FloatType>>> P_kd(M);
        for (uint32_t m = 0u; m < M; m++)
        {
            const VectorType& y_hat_m = y_hat[m];

            std::vector<nanoflann::ResultItem<std::size_t, FloatType>> ret_matches;
            const size_t nMatches =
                xKdTree->radiusSearch(y_hat_m.data(), search_radius, ret_matches);

            if(ret_matches.empty())
            {
                std::size_t                num_results = 5;
                std::vector<std::size_t> ret_index(num_results);
                std::vector<FloatType>    out_dist_sqr(num_results);

                xKdTree->query(
                            y_hat_m.data(), num_results, &ret_index[0], &out_dist_sqr[0]);

                for(uint32_t i = 0u; i < num_results; i++)
                {
                    ret_matches.emplace_back(ret_index[i], out_dist_sqr[i]);
                }

                std::cout << "No mathes were found at m=: " << m << std::endl;
            }

            for(const auto& xNeighbor : ret_matches)
            {
                // calculate p_mn
                FloatType p_mn = calculateP(xNeighbor.second, residual, FloatType(1.0), alpha[m], m_omega);
                P_kd[m].push_back(std::make_pair(xNeighbor.first, p_mn));
            }
        }

        std::vector<FloatType> sumP(N,FloatType(0.0));
        for (uint32_t m = 0u; m < M; m++)
        {
            for(const auto& p : P_kd[m])
            {
                sumP[p.first] += p.second;
            }
        }

        for (uint32_t m = 0u; m < M; m++)
        {
            for(auto& p : P_kd[m])
            {
                p.second /= sumP[p.first];
            }
        }

        // Update nu, N_hat.
        nu.resize(M);
        nu_apo.assign(N, FloatType(0.0));
        N_hat = 0.0;
        for (uint32_t m = 0u; m < M; m++)
        {
            FloatType sumP = 0.0;
            for(const auto p : P_kd[m])
            {
                nu_apo[p.first] += p.second;
                sumP += p.second;
            }
            nu[m] = sumP;
            N_hat += sumP;
        }

        // x_hat
        Px.resize(M);
        x_hat.resize(M);
        for (uint32_t m = 0u; m < M; m++)
        {
            Px[m] = VectorType::Zero();
            for(const auto p : P_kd[m])
            {
                Px[m] += p.second * x[p.first];
            }

            if(nu[m] == FloatType(0.0))
            {
                std::cout << "Is nan at m=: " << m << std::endl;
            }

            x_hat[m] = Px[m] / nu[m];
        }
    }

#else 
    P.resize(N);
    for (uint32_t n = 0u; n < N; n++)
    {
        P[n].resize(M);
    }

    for (uint32_t n = 0u; n < N; n++)
    {
        const VectorType& x_n = x[n];

        FloatType sum_p_mn = 0.0;
        for (uint32_t m = 0u; m < M; m++)
        {
            const VectorType& y_hat_m = y_hat[m];

            const FloatType xyDist2 = (x_n - y_hat_m).squaredNorm();

            // calculate p_mn
            FloatType p_mn = calculateP(xyDist2, residual, sigma(m,m),  alpha[m],  m_omega);

            P[n][m] = p_mn;
            sum_p_mn += p_mn;
        }

        // divide p_mn by the sum.
        for (uint32_t m = 0u; m < M; m++)
        {
            P[n][m] /= m_omega * p_out + sum_p_mn;
        }
    }

    // Update nu, N_hat.
    nu.resize(M);
    nu_apo.assign(N, FloatType(0.0));
    N_hat = 0.0;
    for (uint32_t m = 0u; m < M; m++)
    {
        FloatType sumPn = 0.0;
        for (uint32_t n = 0u; n < N; n++)
        {
            sumPn += P[n][m];
            nu_apo[n] += P[n][m];
        }

        nu[m] = sumPn;
        N_hat += sumPn;
    }

    // x_hat
    x_hat.resize(M);
    for (uint32_t m = 0u; m < M; m++)
    {
        VectorType x_hat_m = VectorType::Zero();
        for (uint32_t n = 0u; n < N; n++)
        {
            x_hat_m += P[n][m] * x[n];
        }
        x_hat[m] = x_hat_m / nu[m];
    }

#endif

    // alpha
    for (uint32_t m = 0u; m < M; m++)
    {
        alpha[m] = std::exp(digamma(m_kappa + nu[m]) - digamma(m_kappa * M + N_hat));
    }

}

/**
 * @brief
 *
 * @tparam FloatType
 */
template <class FloatType, uint32_t dim>
inline void BCPD<FloatType, dim>::MaximizationStep()
{
    TimeTracker tr("MaximizationStep");

    uint32_t N =  x.size();
    uint32_t M =  y.size();

    std::vector<VectorType> E(M);
    std::vector<VectorType> x_hatTrans(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        x_hatTrans[i] = rotation.transpose() * (x_hat[i] - translation) / scale;
        E[i] = (rotation.transpose() * (x_hat[i] - translation)) / scale - y[i];
    }

    // Compute sigma.
    FloatType cc = std::pow(scale, 2.0) / residual;

#if NYSTROM
    Eigen::Vector<FloatType, Eigen::Dynamic> nuVec(M);
    for (uint32_t j = 0u; j < M; j++)
    {
        nuVec[j] = nu[j];
    }
    EigenMatrix S = Q.transpose() * nuVec.asDiagonal() * Q;

    EigenMatrix sigmaKxK = S;
    sigmaKxK += m_lambda / cc * LAMBDA.inverse();
    EigenMatrix sigmaDebug = S * sigmaKxK.inverse();
    sigmaKxK = (EigenMatrix::Identity(kSamples, kSamples) - S * sigmaKxK.inverse());
    sigmaKxK = LAMBDA * (sigmaKxK) / m_lambda;
    //sigmaKxK = LAMBDA * (EigenMatrix::Identity(kSamples, kSamples) - S * sigmaKxK.inverse()) / m_lambda;

    EigenMatrix sigmaKxM = sigmaKxK * Q.transpose();


    // Compute v_hat.
    std::vector<VectorType> v_hatK(kSamples);
    for (uint32_t i = 0u; i < kSamples; i++)
    {
        v_hatK[i] = VectorType::Zero();
        for (uint32_t j = 0u; j < M; j++)
        {
            v_hatK[i] += cc * sigmaKxM(i,j) * nu[j] * E[j];
        }
    }

    // Compute v_hat.
    std::vector<VectorType> v_hat(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        v_hat[i] = VectorType::Zero();
        for (uint32_t j = 0u; j < kSamples; j++)
        {
            v_hat[i] += Q(i,j) * v_hatK[j];
        }
    }

#else
    std::vector<FloatType> diagMat(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        for (uint32_t j = i; j < M; j++)
        {
            sigma(i,j) = (G(i,j) / m_lambda) * std::pow(cc, 2.0) * nu[i] * nu[j];

            if(i == j)
            {
                sigma(i,j) += cc * nu[i];
            }
            else
            {
                sigma(j,i) = sigma(i,j);
            }
        }
    }
    sigma = sigma.inverse();

    for (uint32_t i = 0u; i < M; i++)
    {
        for (uint32_t j = i; j < M; j++)
        {
            sigma(i,j) = -sigma(i,j);

            if(i == j)
            {
                sigma(i,j) += 1.0 / (cc * nu[i]);
            }
            else
            {
                sigma(j,i) = sigma(i,j);
            }
        }
    }

    // In debug mode calculate ref sigma.
#ifndef NDEBUG
    std::cout << "sigma is:" << std::endl << sigma << std::endl;
    
    EigenMatrix Diag(M,M);
    Diag.setIdentity();
    for (uint32_t i = 0u; i < M; i++)
    {
        Diag(i,i) = cc * nu[i];
    }

    EigenMatrix sigmaRef = m_lambda * G.inverse() + Diag;
    sigmaRef = sigmaRef.inverse();
    //sigma = sigmaRef;

    std::cout << "sigma ref is: " << std::endl << sigmaRef << std::endl;

#endif 

    // Compute v_hat.
    std::vector<VectorType> v_hat(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        v_hat[i] = VectorType::Zero();
        for (uint32_t j = 0u; j < M; j++)
        {
            v_hat[i] += cc * sigma(i,j) * nu[j] * E[j];
        }
    }
#endif // NYSTROM

    // Update u_hat.
    std::vector<VectorType> u_hat(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        u_hat[i] = y[i] + v_hat[i];
    }

    std::cout << "u_hat computed" << std::endl;

    // Update s,R,t and residual.
    /* ------------------------------------------------------------------------------------- */
#if 1
    // x_avg
    VectorType x_avg = VectorType::Zero();
    for (uint32_t i = 0u; i < M; i++)
    {
        x_avg += nu[i] * x_hat[i];
    }
    x_avg /= N_hat;

#if 0
    // sigma_avg
    FloatType sigma_avg(0.);
    for (uint32_t i = 0u; i < M; i++)
    {
        sigma_avg += nu[i] * sigma(i,i);
        //std::cout << "sigma(" << i << "," << i << "): " << sigma(i,i) << std::endl;
    }
    sigma_avg /= N_hat;
#endif

    // u_avg
    VectorType u_avg = VectorType::Zero();
    for (uint32_t i = 0u; i < M; i++)
    {
        u_avg += nu[i] * u_hat[i];
    }
    u_avg /= N_hat;

    // Sxu
    MatrixType Sxu = MatrixType::Zero();
    for (uint32_t i = 0u; i < M; i++)
    {
        Sxu += nu[i] * (x_hat[i] - x_avg) * (u_hat[i] - u_avg).transpose();
    }
    Sxu /= N_hat;

    // Suu
    MatrixType Suu =  MatrixType::Zero();
    for (uint32_t i = 0u; i < M; i++)
    {
        Suu += nu[i] * (u_hat[i] - u_avg) * (u_hat[i] - u_avg).transpose();
    }
    Suu /= N_hat;
    //Suu += sigma_avg * MatrixType::Identity();

    // R
    Eigen::JacobiSVD<MatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(Sxu,  Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "Sxu's singular values are:" << std::endl << svd.singularValues() << std::endl;
    auto U = svd.matrixU();
    auto V = svd.matrixV();

    MatrixType R_diag = MatrixType::Identity();
    R_diag(dim- 1u, dim - 1u) = (U * V.transpose()).determinant();
    rotation = U * R_diag * V.transpose();

    std::cout << "Det rot matrix: " << rotation.determinant() << std::endl;

    // scale
    scale = 0.0;
    scale = (rotation.transpose() * Sxu).trace() / Suu.trace();

    // translation
    translation = x_avg - scale * rotation * u_avg;
#endif

    // Update y_hat
    for (uint32_t i = 0u; i < M; i++)
    {
        y_hat[i] = scale * rotation * u_hat[i] + translation;
    }

    // Update residual.
#if NYSTROM

    residual = FloatType(0.0);
    for(uint32_t i = 0u; i < N; i++)
    {
        residual += nu_apo[i] * x[i].squaredNorm();
    }

    for (uint32_t i = 0u; i < M; i++)
    {
        residual -= FloatType(2.0) * Px[i].dot(y_hat[i]);
    }

    for(uint32_t i = 0u; i < M; i++)
    {
        residual += nu[i] * y_hat[i].squaredNorm();
    }

    residual = residual / (FloatType(dim) * N_hat);
    //resudual += std::pow(scale, 2.0) * sigma_avg;

#else
    residual = 0.0;
    for (uint32_t i = 0u; i < M; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            residual += P[j][i] * (y_hat[i] - x[j]).squaredNorm();
        }
    }
    residual /= FloatType(dim) * N_hat;
    //residualDebug += std::pow(scale, 2.0) * sigma_avg;

    residual = residualDebug;
#endif



#if WRITE_DEBUG_OUTPUT

    std::string folder = "Iteration_" + std::to_string(iter);

    std::filesystem::create_directories(outputDir / folder);

    // write x-hat
    auto x_Path = outputDir / folder / "x.ply";
    writePly<FloatType,dim>(x, x_Path);

     auto x_hatPath = outputDir / folder / "x_hat.ply";
    writePly<FloatType,dim>(x_hat, x_hatPath);

     auto yPath = outputDir / folder / "y.ply";
    writePly<FloatType,dim>(y, yPath);

     auto y_hatPath = outputDir / folder / "y_hat.ply";
    writePly<FloatType,dim>(y_hat, y_hatPath);
#endif

#ifndef NDEBUG 

#if PRINT_DEBUG
    // Debug output.
    std::cout << std::endl;
    for(uint32_t i = 0u; i < M; i++)
    {
        std::cout << "x_hat[" << i << "]: " << x_hat[i][0] << ", " << x_hat[i][1] << std::endl;
    }

    std::cout << std::endl;
    for(uint32_t i = 0u; i < M; i++)
    {
        std::cout << "y_hat[" << i << "]: " << y_hat[i][0] << ", " << y_hat[i][1] << std::endl;
    }
#endif

#if VISUALIZE // Visualize output.
    CvPlot::Axes axes = CvPlot::makePlotAxes();

    auto addPlots = [](const std::vector<VectorType>& points, CvPlot::Axes& axesPlot, std::string_view plotType)
    {
        std::vector<cv::Point2d> plotPoints;
        for(const auto& x_i : points)
        {
            plotPoints.emplace_back(x_i[0], x_i[1]);
        }
        axesPlot.create<CvPlot::Series>(plotPoints, plotType.data());
    };

    // Plot x points.
    addPlots(x, axes, "ok");
    //addPlots(y, axes, "ob");

    addPlots(y_hat, axes, "or");    
    //addPlots(x_hat, axes, "or");

    //addPlots(x_hatTrans, axes, "og");
    //addPlots(u_hat, axes, "oy");
    
    CvPlot::show("xPoints", axes);
#endif

#endif
    
}


//template class BCPD<float>;
template class BCPD<double, 2u>;
template class BCPD<double, 3u>;