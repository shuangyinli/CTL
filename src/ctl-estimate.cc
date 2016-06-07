/*
 * ctl-estimate.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */

#include "ctl-estimate.h"

#include <assert.h>

double compute_doc_log_likehood(Document *doc, ctl_model *model) {
    double *log_gamma = doc->log_gamma;
    double *log_phi = model->log_phi;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int doc_num_words = doc->num_words;
    int *doc_words_ptr = doc->words_ptr;
    int *doc_words_cnt_ptr = doc->words_cnt_ptr;
    double log_lik = 0.0;
    for (int i = 0; i < doc_num_words; ++i) {
        int wordid = doc_words_ptr[i];
        double lik_doc = 0.0;
        for (int j = 0; j < num_topics; ++j) {
            lik_doc += doc->words_cnt_ptr[i] * exp(log_gamma[i * num_topics + j]) * exp(log_phi[j * num_words + wordid]);
        }
        log_lik += log(lik_doc);
    }
    return log_lik;
}

double compute_corpus_log_likehood(Document **corpus, ctl_model *model) {
    int num_docs = model->num_docs;
    int log_lik = 0.0;
    for (int d = 0; d < num_docs; ++d) {
        double temp_lik = compute_doc_log_likehood(corpus[d], model);
        corpus[d]->lik = temp_lik;
        log_lik += temp_lik;
        for (int i = 0; i < corpus[d]->num_words; ++ i) {
          double sum = 0.0;
          for (int j = 0; j < model->num_topics; ++ j) {
            sum += exp(corpus[d]->log_gamma[i * model->num_topics + j]);
          }
        }
    }
    for (int i = 0; i < model->num_topics; ++ i) {
      double sum = 0.0;
      for (int j = 0; j < model->num_words; ++ j) {
        sum += exp(model->log_phi[i * model->num_words + j]);
      }
    }
    return log_lik;
}

double compute_doc_log_likehood2(Document *doc, ctl_model *model) {
    double *log_gamma = doc->log_gamma;
    double *log_phi = model->log_phi;
    double *lambda = model->lambda;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int num_labels = model->num_labels;
    int doc_num_labels = doc->num_labels;
    int doc_num_words = doc->num_words;
    int *doc_words_ptr = doc->words_ptr;
    int *doc_words_cnt_ptr = doc->words_cnt_ptr;
    int *doc_labels_ptr = doc->labels_ptr;
    double *doc_u = doc->u;
    double log_lik = 0.0;
    double exp_sum_u = 0.0;
    for (int i = 0; i < doc_num_labels; ++i) {
        int label = doc_labels_ptr[i];
        exp_sum_u += exp(doc_u[label]);
    }

    double *topics = doc->topic;
    double topics_sum = 0.0;
    for (int j = 0; j < num_topics; ++ j) topics[j] = 0.0;
    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        double lambda_sum = 0.0;
        for (int j = 0; j < num_topics; ++ j) lambda_sum += lambda[label * num_topics + j];
        for (int j = 0; j < num_topics; ++ j) {
            topics[j] += exp(doc_u[label]) * lambda[label * num_topics + j] / lambda_sum;
        }
    }
    for (int i = 0; i < num_topics; ++ i) topics_sum += topics[i];
    for (int i = 0; i < num_topics; ++ i) topics[i] /= topics_sum;

    for (int i = 0; i < doc_num_words; ++i) {
        int wordid = doc->words_ptr[i];
        double temp = log(topics[0]) + log_phi[wordid];
        for (int j = 1; j < num_topics; ++j) {
            temp = util::log_sum(temp, log(topics[j]) + log_phi[j * num_words + wordid]);
        }
        log_lik += temp * doc->words_cnt_ptr[i];
    }
    return log_lik;
}

double compute_corpus_log_likehood2(Document **corpus, ctl_model *model) {
    int num_docs = model->num_docs;
    int log_lik = 0.0;
    for (int d = 0; d < num_docs; ++d) {
        double temp_lik = compute_doc_log_likehood2(corpus[d], model);
        corpus[d]->lik = temp_lik;
        log_lik += temp_lik;
    }
    return log_lik;
}
