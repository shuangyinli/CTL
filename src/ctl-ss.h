/*
 * ctl-ss.h
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#ifndef __CTL_SS_H
#define __CTL_SS_H

struct SS {
    double *sigma;
    double *mu;
    int sigma_size_;
    int mu_size_;
    SS(const int &sigma_size, const int &mu_size) : 
        sigma_size_(sigma_size), mu_size_(mu_size) {
        sigma = new double[sigma_size];
        mu = new double[mu_size];
    }
    ~SS() {
        delete []sigma;
        delete []mu;
    }
    void init() {
        memset(sigma, 0, sizeof(double) * sigma_size_);
        memset(mu, 0, sizeof(double) * mu_size_);
    }
};

#endif
