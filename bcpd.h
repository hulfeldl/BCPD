/**
 * @file bcpd.h
 * @brief Bayesian Coherent Point Drift (BCPD) point cloud registration
 * @author Lorenz Hulfeld (lorenz.hulfeld@gmail.com)
 * @version 1.1.0
 * @date 2024-2026
 *
 * @copyright Copyright (c) 2024 Lorenz Hulfeld
 */
#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <KDTreeVectorOfVectorsAdaptor.h>

template <class FloatType, uint32_t dim>
class Kernel
{
    public:
        using VectorType = Eigen::Vector<FloatType, dim>; 
        virtual ~Kernel() = default;
        virtual FloatType compute(const VectorType& x, const VectorType& y) const = 0;
};

/**
 * @brief BCPD Algorithm Implementation
 * 
 */
template <class FloatType, uint32_t dim>
class BCPD
{
    public:
        using VectorType = Eigen::Vector<FloatType, dim>;
        using MatrixType = Eigen::Matrix<FloatType, dim, dim>;
        using EigenVector = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;
        using EigenMatrix = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
        using kdTreeType = KDTreeVectorOfVectorsAdaptor<std::vector<VectorType>, FloatType>;

        static FloatType getLengthSqr(const VectorType& x, const VectorType& y);

        explicit BCPD(FloatType beta,
                      FloatType lambda,
                      FloatType omega,
                      FloatType gamma) noexcept
            : m_beta(beta),
              m_lambda(lambda),
              m_omega(omega),
              m_gamma(gamma)
        {
        }

        explicit BCPD(FloatType beta,
                      FloatType lambda,
                      FloatType omega,
                      FloatType gamma,        
                      FloatType kappa) noexcept
            : m_beta(beta),
              m_lambda(lambda),
              m_omega(omega),
              m_gamma(gamma),        
              m_kappa(kappa)
        {
        }

        ~BCPD() = default;
        BCPD(const BCPD&) = delete;
        BCPD& operator=(const BCPD&) = delete;
        BCPD(BCPD&&) = default;
        BCPD& operator=(BCPD&&) = default;

        void SetInput(const std::vector<VectorType>& x_in,
                      const std::vector<VectorType>& y_in) noexcept
        {
            x = x_in;
            y = y_in;
        }

        void Compute();

    protected:
        void Initialization();
        void ExpectationStep();
        void MaximizationStep();

        void computeExpectationNystrom(uint32_t N, uint32_t M);
        void computeExpectationKdTree(uint32_t N, uint32_t M);
        void computeExpectationDirect(uint32_t N, uint32_t M);

        void computeMaximizationNystrom(uint32_t M, FloatType cc,
                                       const std::vector<VectorType>& E);
        void computeMaximizationDirect(uint32_t M, FloatType cc,
                                      const std::vector<VectorType>& E);

        void updateResidual(uint32_t N, uint32_t M);
        void writeDebugOutput() const;

    private:
        uint32_t iter = 0u;

        // Tuning parameters
        FloatType m_beta = FloatType(2.0);
        FloatType m_lambda = FloatType(2.0);
        FloatType m_omega = FloatType(0.0);
        FloatType m_gamma = FloatType(1.0);        
        FloatType m_kappa = FloatType(1e9);

        static constexpr uint32_t kSamples = 150u;
        static constexpr uint32_t vSamples = 300u;

        // Input Point Clouds
        std::vector<VectorType> x;
        std::vector<VectorType> y;
        std::unique_ptr<kdTreeType> xKdTree;

        // Output
        FloatType scale = FloatType(1.0);
        MatrixType rotation = MatrixType::Identity();
        VectorType translation = VectorType::Zero();
        std::vector<VectorType> y_hat;
        FloatType residual = FloatType(0.0);

        // Initialization
        EigenMatrix G;
        EigenMatrix Q;
        Eigen::DiagonalMatrix<FloatType, Eigen::Dynamic> LAMBDA;
        FloatType sigmaSQR = FloatType(0.0);
        FloatType p_out = FloatType(0.0);
        std::vector<FloatType> alpha;
        std::unique_ptr<Kernel<FloatType, dim>> m_kernel;
        EigenMatrix sigma;

        // Expectation step
        std::vector<FloatType> sgm;
        std::vector<std::vector<FloatType>> P;
        std::vector<VectorType> Px;  
        std::vector<FloatType> nu;
        std::vector<FloatType> nu_apo;
        std::vector<VectorType> x_hat;
        std::vector<VectorType> u_hat;
        FloatType N_hat = FloatType(0.0);
};
