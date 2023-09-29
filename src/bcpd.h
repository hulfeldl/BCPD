/**
 * @file bcpd.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include <array>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <KDTreeVectorOfVectorsAdaptor.h>

template <class FloatType, uint32_t dim>
class Kernel
{
    public:
        using VectorType = Eigen::Vector<FloatType, dim>; 

        virtual FloatType compute(const VectorType& x, const VectorType& y) = 0;
};

/**
 * @brief 
 * 
 */
template <class FloatType,  uint32_t dim>
class BCPD
{
    public:
        using VectorType = Eigen::Vector<FloatType, dim>;
        using MatrixType = Eigen::Matrix<FloatType, dim, dim>;
        using EigenVector = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;
        using EigenMatrix = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
        using kdTreeType = KDTreeVectorOfVectorsAdaptor<std::vector<VectorType>, FloatType>;

        static FloatType getLengthSqr(const VectorType& x, const VectorType& y);

        BCPD(FloatType beta,
            FloatType lambda,
            FloatType omega,
            FloatType gamma)
        : m_beta(beta),
          m_lambda(lambda),
          m_omega(omega),
          m_gamma(gamma)
        {

        }

        BCPD(FloatType beta,
            FloatType lambda,
            FloatType omega,
            FloatType gamma,        
            FloatType kappa)
        : m_beta(beta),
          m_lambda(lambda),
          m_omega(omega),
          m_gamma(gamma),        
          m_kappa(kappa)
        {

        }

        void SetInput(const std::vector<VectorType>& x_in,
                      const std::vector<VectorType> y_in)
        {
            x = x_in;
            y = y_in;
        }
        void Compute();


    protected:
        void Initialization();
        void ExpectationStep();
        void MaximizationStep();

    private:
        // Tuning parameters.
        FloatType m_beta = FloatType(2.0);
        FloatType m_lambda = FloatType(2.0);
        FloatType m_omega = FloatType(0.0);

        FloatType m_gamma = FloatType(1.0);        
        FloatType m_kappa = FloatType(1e9);

        // Input Point Clouds.
        std::vector<VectorType> x; // target point cloud
        std::vector<VectorType> y; // source point cloud, deformable
        std::unique_ptr<kdTreeType> xKdTree;

        // Output
        FloatType scale;
        MatrixType rotation;
        VectorType translation;

        std::vector<VectorType> y_hat;

        FloatType residual;

        // Initialization.
         EigenMatrix G;
        FloatType sigmaSQR;
        FloatType p_out = FloatType(0.0);
        std::vector<FloatType> alpha;
        std::unique_ptr<Kernel<FloatType,dim>> m_kernel;
        EigenMatrix sigma;

        // Expectation setp.
        std::vector<FloatType> sgm;
        std::vector<std::vector<FloatType>> P;
        std::vector<FloatType> nu;
        std::vector<FloatType> nu_apo;
        FloatType N_hat;
        std::vector<VectorType> x_hat;
};

