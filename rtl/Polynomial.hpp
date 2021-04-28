#ifndef __RTL_POLYNOMIAL__
#define __RTL_POLYNOMIAL__

#include "Base.hpp"
#include "Line.hpp"
#include <vector>
#include <algorithm>
#include <random>

class Polynomial 
{
public:
    Polynomial(std::vector<double> _coeffs) : coeffs(_coeffs) { }

    std::vector<double> coeffs;
};

class PolynomialEstimator : virtual public RTL::Estimator<Polynomial, Point, std::vector<Point> > 
{
public:
    virtual Polynomial ComputeModel(const std::vector<Point>& data, const std::set<int>& samples)
    {
        // There are three schemes. Here we use Lagrange Interpolation 
        // Ref. http://bueler.github.io/M310F11/polybasics.pdf
        size_t M = samples.size();
        std::vector<double> res(M, 0);
        for (auto itr1 = samples.begin(); itr1 != samples.end(); itr1++) {
            const Point& current_point = data[*itr1];
            std::vector<double> tmp_coeffs(M, 0);
            // Start with a constant polynomial
            tmp_coeffs[0] = current_point.y;
            double prod = 1;
            for(auto itr2 = samples.begin(); itr2 != samples.end(); itr2++) {
                const Point& point = data[*itr2];
                if (current_point.x == point.x) continue;
                prod *= current_point.x - point.x;
                double precedent = 0;
                for (auto resptr = tmp_coeffs.begin(); resptr < tmp_coeffs.end(); resptr++) {
                    // Compute the new coefficient of X^i based on
                    // the old coefficients of X^(i-1) and X^i
                    double newres = (*resptr) * (-point.x) + precedent;
                    precedent = *resptr;
                    *resptr = newres;
                }
            }
            std::transform(
                res.begin(), res.end(), 
                tmp_coeffs.begin(), res.begin(), 
                [=] (double old_coeff, double add) { return old_coeff + add / prod; } 
                );     
        }
        Polynomial polynomial(res);

        return polynomial;
    }

    virtual double ComputeError(const Polynomial& polynomial, const Point& point)
    {
        double err = point.y;
        for (int i = 0; i < polynomial.coeffs.size(); i++) {
            err -= polynomial.coeffs[i] * pow(point.x, i);
        }
        return err;
    }
};

class PolynomialObserver : virtual public RTL::Observer<Polynomial, Point, std::vector<Point> >
{
public:
    PolynomialObserver(Point _max = Point(640, 480), Point _min = Point(0, 0)) : RANGE_MAX(_max), RANGE_MIN(_min) { }

    virtual std::vector<Point> GenerateData(const Polynomial& poly, int N, std::vector<int>& inliers, double noise = 0, double ratio = 1)
    {
        std::mt19937 generator;
        std::uniform_real_distribution<double> uniform(0, 1);
        std::normal_distribution<double> normal(0, 1);

        std::vector<Point> data;

        for (int i = 0; i < N; i++)
        {
            Point point;
            point.x = (RANGE_MAX.x - RANGE_MIN.x) * uniform(generator) + RANGE_MIN.x;
            double vote = uniform(generator);
            if (vote > ratio)
            {
                // Generate an outlier
                point.y = (RANGE_MAX.y - RANGE_MIN.y) * uniform(generator) + RANGE_MIN.y;
            }
            else
            {
                // Generate an inlier
                double y = 0;
                for (int i = 0; i < poly.coeffs.size(); i++) {
                    y += poly.coeffs[i] * pow(point.x, i);
                }
                point.y = y;
                point.x += noise * normal(generator);
                point.y += noise * normal(generator);
                inliers.push_back(i);
            }
            data.push_back(point);
        }
        
        return data;
    }

    const Point RANGE_MIN;

    const Point RANGE_MAX;
};

#endif // End of '__RTL_POLYNOMIAL__'