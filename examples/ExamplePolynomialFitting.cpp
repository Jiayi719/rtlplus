#include "RTL.hpp"
#include <iostream>

using namespace std;

// The main function
int main(void)
{
    // Generate noisy data from the truth
    std::vector<double> coeffs({1.0, 2.0, 1.0});
    Polynomial trueModel(coeffs);
    vector<int> trueInliers;
    PolynomialObserver observer;
    vector<Point> data = observer.GenerateData(trueModel, 100, trueInliers, 0., 0.6);
    if (data.empty()) return -1;

    // Find the best model using RANSAC
    PolynomialEstimator estimator;
    RTL::RANSAC<Polynomial, Point, vector<Point> > ransac(&estimator);
    std::vector<double> res(coeffs.size(), 0.);
    Polynomial model(res);
    double loss = ransac.FindBest(model, data, data.size(), 3);

    // Determine inliers using the best model if necessary
    vector<int> inliers = ransac.FindInliers(model, data, data.size());

    // Print the result
    cout << "- True Model:  ";
    for (auto coeff : trueModel.coeffs) {
        cout << coeff << ", ";
    }
    cout << endl;
    cout << "- Found Model: ";
    for (auto coeff : model.coeffs) {
        cout << coeff << ", ";
    }
    cout << endl;
    cout << " (Loss: " << loss << ")" << endl;
    cout << "- The Number of Inliers: " << inliers.size() << " (N: " << data.size() << ")" << endl;

    return 0;
}