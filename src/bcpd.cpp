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
#include <iostream>
#include <CvPlot/cvplot.h>
#include "nanoflann.hpp"
//#include <eigen3/Eigen/Dense>
//#include <armadillo>

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


        FloatType compute(const Kernel<FloatType, dim>::VectorType& x, const Kernel<FloatType, dim>::VectorType& y) override
        {
            return std::exp( -(x - y).dot(x - y) / (2 * h *h)); 
        }

    private:
        FloatType h = 0.5;
};


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
    
    while(residual > c_resError)
    {
        ExpectationStep();
        MaximizationStep();
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

    for (uint32_t m = 0; m < M; m++)
    {
        for (uint32_t n = 0; n < N; n++)
        {
            residual += (x[n] - y[m]).dot(x[n] - y[m]);
        }
    }
    
    residual *= (m_gamma * m_gamma)  / FloatType(M * N * dim);

    // sigma
    sigma.resize(M,M);
    sigma.setIdentity();


    G.resize(M,M);
    /* G */
    for (uint32_t i = 0; i < M; i++)
    {
        for (uint32_t j = i; j < M; j++)
        {
            G(i, j) = G(j,i) = m_kernel->compute(y[i], y[j]);
        }
    }

    // Create kd-tree.
    xKdTree = std::make_unique<kdTreeType>(dim /*dim*/, x, 10 /* max leaf */);

}

/**
 * @brief
 *
 * @tparam FloatType
 */
template <class FloatType, uint32_t dim>
inline void BCPD<FloatType, dim>::ExpectationStep()
{
    uint32_t N =  x.size();
    uint32_t M =  y.size();

    sigmaSQR = residual;

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

    // Calculate P using kdTree.
    std::vectorP.resize(N);
    for (uint32_t m = 0u; m < M; m++)
    {
        const VectorType& y_hat_m = y_hat[m];

        std::vector<nanoflann::ResultItem<std::size_t, FloatType>> ret_matches;
        FloatType search_radius = FloatType(10 * 10) * residual;
const size_t nMatches =
            xKdTree->radiusSearch(y_hat_m.data(), search_radius, ret_matches);

        for(const auto& xNeighbor : ret_matches)
        {
                        // calculate p_mn
            FloatType p_mn = calculateP(xNeighbor.second, residual, sigma(m,m), alpha[m], m_omega);
        }
    }

#if 0
    // Debug, stupid identiy matrix simulate ICP.
    for (uint32_t n = 0u; n < N; n++)
    {
        FloatType sum_p_mn = 0.0;
        for (uint32_t m = 0u; m < M; m++)
        {
            P[n][m] = std::exp( -(x[n] - y_hat[m]).squaredNorm() / (2.0 * sigmaSQR) );
            sum_p_mn += P[n][m];
        }

        // divide p_mn by the sum.
        for (uint32_t m = 0u; m < M; m++)
        {
            P[n][m] /= omega * p_out + sum_p_mn;
        }
    }
#endif


        // divide p_mn by the sum.
        for (uint32_t m = 0u; m < M; m++)
        {
            std::cout << "P(" << m << "," << m << "): " << P[m][m] << std::endl;
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

    // In debug mode calculate ref v.
#ifndef NDEBUG
    std::cout << "sigma is:" << std::endl << sigma << std::endl;
    
    //auto A = Diag.inverse() / lambda + G;
    //A.colPivHouseholderQr().solve(E);

    EigenMatrix A = sigma * Diag;

    std::vector<VectorType> v_hatRef(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        v_hatRef[i] = VectorType::Zero();
        for (uint32_t j = 0u; j < M; j++)
        {
            v_hatRef[i] += A(i,j) * E[j];
        }
    }

#endif

    // Update u_hat.
    std::vector<VectorType> u_hat(M);
    for (uint32_t i = 0u; i < M; i++)
    {
        u_hat[i] = y[i] + v_hat[i];
    }

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

    // r_avg
    FloatType sigma_avg(0.);
    for (uint32_t i = 0u; i < M; i++)
    {
        sigma_avg += nu[i] * sigma(i,i);
        std::cout << "sigma(" << i << "," << i << "): " << sigma(i,i) << std::endl;
    }
    sigma_avg /= N_hat;

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
    std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
    auto U = svd.matrixU();
    auto V = svd.matrixV();

    MatrixType R_diag = MatrixType::Identity();
    R_diag(dim- 1u, dim - 1u) = (U * V.transpose()).determinant();

    rotation = U * R_diag * V.transpose();
    //rotation.transposeInPlace();

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
    residual = FloatType(0.0);
    for(uint32_t i = 0u; i < N; i++)
    {
        residual += nu_apo[i] * x[i].dot(x[i]);
    }

    for (uint32_t i = 0u; i < M; i++)
    {
        VectorType row = VectorType::Zero();
        for (uint32_t j = 0; j < N; j++)
        {
            row += P[j][i] * x[j];
        }
        residual -= FloatType(2.0) * row.dot(y_hat[i]);
    }

    for(uint32_t i = 0u; i < M; i++)
    {
        residual += nu[i] * y_hat[i].dot(y_hat[i]);
    }


    residual = residual / (FloatType(dim) * N_hat);
    //resudual += std::pow(scale, 2.0) * sigma_avg;

    // Debug
    FloatType residualDebug = 0.0;
    for (uint32_t i = 0u; i < M; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            residualDebug += P[j][i] * (y_hat[i] - x[j]).squaredNorm();
        }
    }
    residualDebug /= FloatType(dim) * N_hat;
    //residualDebug += std::pow(scale, 2.0) * sigma_avg;

    residual = residualDebug;


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

    #ifndef NDEBUG 
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
    
    }


//template class BCPD<float>;
template class BCPD<double, 2u>;
template class BCPD<double, 3u>;