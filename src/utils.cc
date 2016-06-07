/*
 * ctl-utils.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */

#include "utils.h"

#include <math.h>
#include <assert.h>
#include <stdlib.h>

#include <random>
#include <iostream>

#include <armadillo>

double util::random_from_gamma(const double &alpha) {
  std::default_random_engine generator;
  std::gamma_distribution<double> distribution(alpha, 1.0);
  return distribution(generator);
}

void util::random_from_dirichlet(double *from, double *sample, const int &size) {
  for (int i = 0; i < size; ++ i) sample[i] = random_from_gamma(from[i]);
  double sum = 0.0;
  for (int i = 0; i < size; ++ i) sum += sample[i];
  for (int i = 0; i < size; ++ i) sample[i] /= sum;
}

void util::get_matrix_inverse(double* in_matrix, double* out_matrix, const int &length, const int &height) {
  const int kDoubleLength = 300;
  const double kEps = 1e-6;
  arma::Mat<double> in(length, height);
  char str[kDoubleLength];
  for (int i = 0; i < length; ++ i) {
    for (int j = 0; j < height; ++ j) {
      sprintf(str, "%.6lf", in_matrix[i * height + j]);
      sscanf(str, "%lf", &in(i, j));
    }
  }
  for (int i = 0; i < length; ++ i) in(i, i) += kEps * (rand() % 9 + 1);
  for (int i = 0; i < length; ++ i) {
    for (int j = 0; j < i; ++ j) {
      in(i, j) = in(j, i) = (in(i, j) + in(j, i)) * 0.5;
    }
  }
  arma::Mat<double> out = inv(in);
  for (int i = 0; i < length; ++ i) {
    for (int j = 0; j < height; ++ j) {
      out_matrix[i * height + j] = out(i, j);
    }
  }
}
