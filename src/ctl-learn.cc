/*
 * ctl-learn.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#include "ctl-learn.h"
#include "utils.h"

#include <assert.h>

void normalize_log_matrix_rows(double* log_mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double temp = log_mat[ i * cols];
        for (int j = 1; j < cols; j++) temp = util::log_sum(temp, log_mat[i * cols + j]);
        for (int j = 0; j < cols; j++) log_mat[i*cols + j] -= temp;
    }
}

void learn_phi(Document** corpus, ctl_model* model) {
    int num_docs = model->num_docs;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int num_labels = model->num_labels;
    bool* reset_phi_flag = new bool[num_topics * num_words];
    memset(reset_phi_flag, 0, sizeof(bool) * num_topics * num_words);
    for (int d = 0; d < num_docs; d++) {
        Document* doc = corpus[d];
        int doc_num_labels = doc->num_labels;
        int doc_num_words = doc->num_words;
        for (int k = 0; k < num_topics; k++) {
            for (int i = 0; i < doc_num_words; i++) {
                int wordid = doc->words_ptr[i];
                if (!reset_phi_flag[k * num_words + wordid]) {
                    reset_phi_flag[k * num_words + wordid] = true;
                    model->log_phi[k * num_words + wordid] = log(doc->words_cnt_ptr[i]) + doc->log_gamma[i*num_topics + k];
                }
                else {
                    model->log_phi[k * num_words + wordid] = util::log_sum(model->log_phi[k * num_words + wordid], doc->log_gamma[i*num_topics + k] + log(doc->words_cnt_ptr[i]));
                }
                if (isnan(model->log_phi[k * num_words + wordid])) {
                    printf("gamma %d %lf %lf ",d, doc->log_gamma[i*num_topics +k], model->log_phi[k * num_words + wordid]);
                }
            }
        }
    }
    normalize_log_matrix_rows(model->log_phi, num_topics, model->num_words);
    delete[] reset_phi_flag;
}

void learn_mu(ctl_model* model, SS* ss) {
    double* mu = model->mu;
    int num_labels = model->num_labels;
    int num_docs = model->num_docs;
    for (int i = 0; i < num_labels; ++ i) mu[i] = ss->mu[i] / num_docs;
}

void learn_sigma(ctl_model* model, SS* ss) {
    double* sigma = model->sigma;
    double* mu = model->mu;
    int num_labels = model->num_labels;
    int num_docs = model->num_docs;
    for (int i = 0; i < num_labels * num_labels; ++ i) sigma[i] = 0.0;
    for (int i = 0; i < num_labels; ++ i) {
        for (int j = 0; j < num_labels; ++ j) {
            sigma[i * num_labels + j] = 1.0 / num_docs  
                * (ss->sigma[i * num_labels + j] + mu[i] * mu[j]* num_docs - ss->mu[i] * mu[j] - ss->mu[j] * mu[i]);
        }
    }
    util::get_matrix_inverse(sigma, model->inv_sigma, num_labels, num_labels);
}
