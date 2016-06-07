/*
 * ctl-inference.h
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#ifndef CTL_INFERENCE_H
#define CTL_INFERENCE_H

#include "utils.h"
#include "ctl.h"
#include "pthread.h"
#include "unistd.h"
#include "stdlib.h"
#include "ctl-estimate.h"
#include "ctl-ss.h"

struct GslBundle {
    Document* doc;
    ctl_model* model;
    GslBundle(Document* doc_ = NULL, ctl_model* model_ = NULL) : doc(doc_), model(model_) {
    }
    ~GslBundle() {
        doc = NULL;
        model = NULL;
    }
};

struct Thread_Data {
    Document** corpus;
    int start;
    int end;
    Config* config;
    ctl_model* model;
    SS* ss;
    Thread_Data(Document** corpus_, int start_, int end_, Config* config_, ctl_model* model_, SS* ss_) : corpus(corpus_), start(start_), end(end_), config(config_), model(model_), ss(ss_) {
    }
};

void inference_gamma(Document* doc, ctl_model* model, double* lambda);
void run_thread_inference(Document** corpus, ctl_model* model, Config* config, SS* ss);
void do_lda_inference(Document** corpus, ctl_model* model, Config* config);
void sample_lambda_from_Lambda(double* Lambda, double* lambda, const int &num_topics, const int &num_labels);
void inference_Lambda(double* Lambda, double* lambda, const int &num_topics, const int &num_labels, const int &level);
void inference_lambda(Document** corpus, ctl_model* model, Config* config);
void init_lambda(double* lambda, const int &size);
void init_sigma_mu_phi(ctl_model* model);
void inference_delta(Document* doc, ctl_model* model, Config* config);
void inference_u(Document* doc, ctl_model* model, Config* config);

#endif
