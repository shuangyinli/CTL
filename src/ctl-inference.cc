/*
 * ctl-inference.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#include "ctl-inference.h"

#include <assert.h>

#include "ctl-ss.h"
#include "ctl-estimate.h"
#include "config.h"
#include "utils.h"
#include "params.h"
#include "gsl-wrappers.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

extern pthread_mutex_t mutex;
extern llna_params PARAMS;

void inference_lambda_l(Document** corpus, ctl_model* model, Config* config, int index);
void inference_delta_i(Document* doc, ctl_model* model, Config* config, const int &index);
void fdf_u(const gsl_vector* p, void* params, double* f, gsl_vector* df);
void df_u(const gsl_vector* p, void* params, gsl_vector* df);
double f_u(const gsl_vector* p, void* params);
double get_u_function(Document* doc, ctl_model* model);

void inference_alpha_beta(Document* doc, ctl_model* model) {
    int doc_num_words = doc->num_words;
    int num_topics = model->num_topics;
    int doc_num_labels = doc->num_labels;
    int* doc_labels_ptr = doc->labels_ptr;
    int* doc_words_cnt_ptr = doc->words_cnt_ptr;
    double* doc_u = doc->u;
    double* doc_delta = doc->delta;
    double* lambda = model->lambda;
    double* lambda_sum = new double[doc_num_labels];
    double* log_gamma = doc->log_gamma;

    for (int i = 0; i < doc_num_labels; ++ i) {
        lambda_sum[i] = 0.0;
        for (int j = 0; j < num_topics; ++ j) {
            int label = doc_labels_ptr[i];
            lambda_sum[i] += lambda[label * num_topics + j];
        }
    }
    //Compute Alpha
    doc->alpha = 0.0;
    double gamma_sum = 0.0;
    int all_word_cnt = 0;
    for (int n = 0; n < doc_num_words; ++ n) {
        for (int k = 0; k < num_topics; ++ k) {
            for (int i = 0; i < doc_num_labels; ++ i) {
                int label = doc_labels_ptr[i];
                doc->alpha += exp(log_gamma[n * num_topics + k]) * doc_words_cnt_ptr[n] * 
                              exp(doc_u[label] + 0.5 * doc_delta[label]) * lambda[label * num_topics + k] / lambda_sum[i];
                //--Exception!!
                if (isinf(doc->alpha)) {
                    puts("alpha inf");
                    printf("%lf %lf %lf %lf %lf\n", log_gamma[n * num_topics + k], doc_u[label], doc_delta[label], lambda[label * num_topics + k], lambda_sum[i]);
                    char ch = getchar();
                }
                //--
            }
            gamma_sum += exp(log_gamma[n * num_topics + k]) * doc_words_cnt_ptr[n];
        }
        all_word_cnt += doc_words_cnt_ptr[n];
    }
    assert(fabs(all_word_cnt - gamma_sum) < 1e-4);
    doc->alpha /= gamma_sum;
    //Compute Beta
    doc->beta = 0.0;
    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        doc->beta += exp(doc_u[label] + 0.5 * doc_delta[label]);
    }
}

void init_sigma_mu_phi(ctl_model* model) {
    int num_topics = model->num_topics;
    int num_labels = model->num_labels;
    int num_words = model->num_words;
    double* sigma = model->sigma;
    double* mu = model->mu;
    double* log_phi = model->log_phi;
    double* inv_sigma = model->inv_sigma;

    memset(sigma, 0, sizeof(double) * num_labels * num_labels);
    for (int i = 0; i < num_labels; ++ i) {
        sigma[i * num_labels + i] = 1.0;
    }
    util::get_matrix_inverse(sigma, inv_sigma, num_labels, num_labels);
    memset(mu, 0, sizeof(double) * num_labels);
    for (int i = 0; i < num_topics; ++ i) {
        double sum = 0.0;
        for (int j = 0; j < num_words; ++ j) {
            log_phi[i * num_words + j] = util::random();
            sum += log_phi[i * num_words + j];
        }
        for (int j = 0; j < num_words; ++ j) {
            log_phi[i * num_words + j] = log(log_phi[i * num_words + j] / sum);
        }
    }
}

int converged (double *u, double *v, int n, double threshold) {
    /* return 1 if |a - b|/|a| < threshold */
    double us = 0;
    double ds = 0;
    double d;
    int i;

    for (i = 0; i < n; i++)
        us += u[i] * u[i];

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        ds += d * d;
    }

    if (sqrt(ds / us) < threshold) return 1;
    else return 0;
}

//Newton methods for Lambda
void inference_Lambda(double* Lambda, double* lambda, const int &num_topics, const int &num_labels, const int &level) {

    int M = num_labels;
    int K = num_topics;

    int i, j, t;
    double *g, *h, *pg, *palpha;
    double z, sh, hgz;
    double psg, spg, gs;
    double alpha0, palpha0;

    double* rous = new double[M * K];

    for(i=0; i<M; i++){
        for(j=0; j<K; j++){
            rous[i*K + j] = lambda[i * K + j];
        }
    }

    /* allocate arrays */
    if ((g = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate g.\n");
        return;
    }
    if ((h = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate h.\n");
        return;
    }
    if ((pg = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate pg.\n");
        return;
    }
    if ((palpha = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate palpha.\n");
        return;
    }

    /* initialize */
    if (level == 0)
    {
        for (i = 0; i < K; i++) {
            for (j = 0, z = 0; j < M; j++)
                z += rous[j*K+i];
            Lambda[i] = z / (M * K);
        }
    } else {
        for (i = 0; i < K; i++) {
            for (j = 0, z = 0; j < M; j++)
                z += rous[j*K+i];
            Lambda[i] = z / (M * K * pow(10, level));
        }
    }
    psg = 0;
    for (i = 0; i < M; i++) {
        for (j = 0, gs = 0; j < K; j++)
            gs += rous[i*K+j];
        psg += util::digamma(gs);
    }
    for (i = 0; i < K; i++) {
        for (j = 0, spg = 0; j < M; j++)
            spg += util::digamma(rous[j*K+i]);
        pg[i] = spg - psg;
    }

    /* main iteration */
    for (t = 0; t < MAX_NEWTON_ITERATION; t++)
    {
        for (i = 0, alpha0 = 0; i < K; i++)
            alpha0 += Lambda[i];
        palpha0 = util::digamma(alpha0);

        for (i = 0; i < K; i++)
            g[i] = M * (palpha0 - util::digamma(Lambda[i])) + pg[i];
        for (i = 0; i < K; i++)
            h[i] = - 1 / util::trigamma(Lambda[i]);
        for (i = 0, sh = 0; i < K; i++)
            sh += h[i];

        for (i = 0, hgz = 0; i < K; i++)
            hgz += g[i] * h[i];
        hgz /= (1 / util::trigamma(alpha0) + sh);

        for (i = 0; i < K; i++)
            Lambda[i] = Lambda[i] - h[i] * (g[i] - hgz) / M;

        for (i = 0; i < K; i++)
            if (Lambda[i] < 0) {
                if (level >= MAX_RECURSION_LIMIT) {
                    fprintf(stderr, "newton:: maximum recursion limit reached.\n");
                    exit(1);
                } else {
                    free(g);
                    free(h);
                    free(pg);
                    free(palpha);
                    return inference_Lambda(Lambda, lambda, num_topics, num_labels, 1 + level);
                }
            }

        if ((t > 0) && converged(Lambda, palpha, K, 1.0e-4)) {
            free(g);
            free(h);
            free(pg);
            free(palpha);
            return;
        } else
            for (i = 0; i < K; i++)
                palpha[i] = Lambda[i];

    }
    fprintf(stderr, "newton:: maximum iteration reached. t = %d\n", t);

    delete[] rous;
    return;

}

void init_lambda(double* lambda, const int &size) {
    for (int i = 0; i < size; ++ i) {
        lambda[i] = util::random();
    }
}

void sample_lambda_from_Lambda(double* Lambda, double* lambda, const int &num_topics, const int &num_labels) {
    for (int i = 0; i < num_labels; ++ i) {
        util::random_from_dirichlet(Lambda, lambda + i * num_topics, num_topics);
    }
}

double d2f_delta_i(const double &delta_i, const int &index, Document* doc, ctl_model* model) {
    double* lambda = model->lambda;
    int doc_num_topics = doc->num_topics;
    int doc_num_words = doc->num_words;
    int* words_cnt_ptr = doc->words_cnt_ptr;
    double *log_gamma = doc->log_gamma;
    double *u = doc->u;
    double alpha = doc->alpha;
    double beta = doc->beta;

    double lambda_sum = 0.0;
    for (int k = 0; k < doc_num_topics; ++ k) lambda_sum += lambda[index * doc_num_topics + k];
    double sum = 0.0;
    for (int n = 0; n < doc_num_words; ++ n) {
        for (int k = 0; k < doc_num_topics; ++ k) {
            sum += 0.25 * words_cnt_ptr[n] * exp(log_gamma[n * doc_num_topics + k]) / alpha * lambda[index * doc_num_topics + k] / lambda_sum * exp(u[index] + 0.5 * delta_i);
        }
    }
    int all_word_cnt = 0;
    for (int i = 0; i < doc_num_words; ++ i) all_word_cnt += words_cnt_ptr[i];
    double res = sum - 0.25 * all_word_cnt / beta * exp(u[index] + 0.5 * delta_i) - 0.5 / delta_i / delta_i;
    return res;
}

double df_delta_i(const double &delta_i, const int &index, Document* doc, ctl_model* model) {
    double *lambda = model->lambda;
    double* sigma = model->sigma;
    int num_labels = model->num_labels;
    int doc_num_topics = doc->num_topics;
    int doc_num_words = doc->num_words;
    int* words_cnt_ptr = doc->words_cnt_ptr;
    double* inv_sigma = model->inv_sigma;
    double *log_gamma = doc->log_gamma;
    double *u = doc->u;
    double alpha = doc->alpha;
    double beta = doc->beta;
    double lambda_sum = 0.0;

    for (int i = 0; i < doc_num_topics; ++ i) lambda_sum += lambda[index * doc_num_topics + i];
    double sum = 0.0;
    for (int n = 0; n < doc->num_words; ++ n) {
        for (int k = 0; k < doc->num_topics; ++ k) {
            sum += 0.5 * words_cnt_ptr[n] * exp(log_gamma[n * doc_num_topics + k]) / alpha * lambda[index * doc_num_topics + k] / lambda_sum * exp(u[index] + 0.5 * delta_i);
        }
    }
    int all_word_cnt = 0;
    for (int i = 0; i < doc_num_words; ++ i) all_word_cnt += words_cnt_ptr[i];
    double res = -0.5 * inv_sigma[index * num_labels + index] + sum - all_word_cnt * 0.5 / beta * exp(u[index] + 0.5 * delta_i) + 0.5 / delta_i;

    return res;
}

void inference_gamma(Document* doc, ctl_model* model) {
    double* lambda = model->lambda;
    double* log_phi = model->log_phi;
    int* doc_words_cnt_ptr = doc->words_cnt_ptr;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int doc_num_words = doc->num_words;
    int doc_num_labels = doc->num_labels;
    int* labels_ptr = doc->labels_ptr;
    double* delta = doc->delta;
    double* log_gamma = doc->log_gamma;
    double* u = doc->u;
    double alpha = doc->alpha;

    double* lambda_sum = new double[doc_num_labels];
    double* sum = new double[num_topics];

    for (int i = 0; i < doc_num_labels; ++ i) lambda_sum[i] = 0;
    for (int i = 0; i < doc_num_labels; ++ i) {
        for (int k = 0; k < num_topics; ++ k) {
            int label = labels_ptr[i];
            lambda_sum[i] += lambda[label * num_topics + k];
        }
    }
    for (int k = 0; k < num_topics; k++) {
        double temp = log(alpha);
        for (int i = 0; i < doc_num_labels; i++) {
            int label = labels_ptr[i];
            temp += 1.0 / alpha * exp(u[label] + 0.5 * delta[label]) * lambda[label * num_topics + k] / lambda_sum[i];
        }
        sum[k] = temp;
        //--Exception!!
        if (isnan(sum[k])) {
            for (int i = 0; i < doc_num_labels; ++ i) {
                int label = labels_ptr[i];
                printf("%lf %lf %lf %lf %lf\n", alpha, u[label], delta[label], lambda[label * num_topics + k], lambda_sum[i]);
                char ch = getchar();
            }
        }
        //--
    }
    for (int i = 0; i < doc_num_words; i++) {
        int wordid = doc->words_ptr[i];
        double sum_log_gamma = 0;
        for (int k = 0; k < num_topics; k++) {
            double temp = log_phi[k * num_words + wordid] * doc_words_cnt_ptr[i] + sum[k];
            log_gamma[i * num_topics + k] = temp;
            if (k == 0) sum_log_gamma = temp;
            else sum_log_gamma = util::log_sum(sum_log_gamma, temp);
        }
        for (int k = 0; k < num_topics; k++) log_gamma[i*num_topics + k] -= sum_log_gamma;
    }

    for (int i = 0; i < doc_num_words; ++ i) {
      double sum = 0.0;
      for (int j = 0; j < num_topics; ++ j) {
        sum += exp(log_gamma[i * num_topics + j]);
      }
    }
    delete []sum;
    delete []lambda_sum;
}

void get_descent_lambda_l(Document** corpus, ctl_model* model, double* descent_lambda_l, const int &index) {
    int num_docs = model->num_docs;
    int num_topics = model->num_topics;
    double* Lambda = model->Lambda;
    double* lambda_l = model->lambda + index * num_topics;
    double lambda_l_sum = 0.0;
    for (int i = 0; i < num_topics; ++ i) lambda_l_sum += lambda_l[i];
    for (int i = 0; i < num_topics; ++ i) {
        descent_lambda_l[i] = (Lambda[i] - lambda_l[i]) * (util::trigamma(lambda_l[i]) - util::trigamma(lambda_l_sum));
        for (int d = 0; d < num_docs; ++ d) {
            Document* doc = corpus[d];
            int doc_num_words = doc->num_words;
            int doc_num_labels = doc->num_labels;
            int* doc_words_cnt_ptr = doc->words_cnt_ptr;
            int* doc_labels_ptr = doc->labels_ptr;
            double* doc_log_gamma = doc->log_gamma;
            double* doc_u = doc->u;
            double* doc_delta = doc->delta;
            double alpha = doc->alpha;
            double exp_value = exp(doc_u[index] + doc_delta[index] * 0.5);
            bool exist = false;
            for (int j = 0; j < doc_num_labels && !exist; ++ j) {
                if (doc_labels_ptr[j] == index) exist = true;
            }
            if (!exist) continue;
            for (int n = 0; n < doc_num_words; ++ n) {
                descent_lambda_l[i] += doc_words_cnt_ptr[n] * exp(doc_log_gamma[n * num_topics + i]) / alpha * exp_value * (lambda_l_sum - lambda_l[i]) / (lambda_l_sum * lambda_l_sum);
            }
        }
    }
}

double get_lambda_l_function(Document** corpus, ctl_model* model, const int &index) {
    int num_topics = model->num_topics;
    int num_docs = model->num_docs;
    double* Lambda = model->Lambda;
    double* lambda_l = model->lambda + index * num_topics;
    double lambda_l_sum = 0.0;
    double res = 0.0;
    for (int i = 0; i < num_topics; ++ i) lambda_l_sum += lambda_l[i];
    for (int i = 0; i < num_topics; ++ i) {
        res += (Lambda[i] - lambda_l[i]) * (util::digamma(lambda_l[i]) - util::digamma(lambda_l_sum)) + util::log_gamma(lambda_l[i]);
    }
    for (int d = 0; d < num_docs; ++ d) {
        Document* doc = corpus[d];
        int doc_num_words = doc->num_words;
        int doc_num_labels = doc->num_labels;
        int* doc_words_cnt_ptr = doc->words_cnt_ptr;
        int* doc_labels_ptr = doc->labels_ptr;
        double* doc_log_gamma = doc->log_gamma;
        double* doc_u = doc->u;
        double* doc_delta = doc->delta;
        double alpha = doc->alpha;
        double exp_value = exp(doc_u[index] + doc_delta[index] * 0.5);
        bool exist = false;
        for (int i = 0; i < doc_num_labels && !exist; ++ i) {
            if (doc_labels_ptr[i] == index) exist = true;
        }
        if (!exist) continue;
        for (int n = 0; n < doc_num_words; ++ n) {
            for (int k = 0; k < num_topics; ++ k) {
                res += doc_words_cnt_ptr[n] * exp(doc_log_gamma[n * num_topics + k]) / alpha * exp_value * lambda_l[k] / lambda_l_sum;
            }
        }
    }
    return res;
}

inline void init_lambda_l(double* lambda_l,int num_topics) {
    for (int i = 0; i < num_topics; i++) lambda_l[i] = util::random();//init 100?!
}

inline bool has_neg_value(double* vec,int dim) {
    for (int i =0; i < dim; i++) {
        if (vec[dim] < 0)return true;
    }
    return false;
}

void inference_lambda(Document** corpus, ctl_model* model, Config* config) {
    double* lambda = model->lambda;
    for (int l = 0; l < model->num_labels; ++ l) {
        inference_lambda_l(corpus, model, config, l);
    }
}

void inference_lambda_l(Document** corpus, ctl_model* model, Config* config, int index) {
    int num_topics = model->num_topics;
    double* descent_lambda_l = new double[num_topics];
    double* lambda_l = model->lambda + index * num_topics;
    init_lambda_l(lambda_l, num_topics);
    double z = get_lambda_l_function(corpus, model, index);
    double learn_rate = config->lambda_learn_rate;
    double eps = 10000;
    int num_round = 0;
    int max_lambda_iter = config->lambda_max_iter;
    double lambda_min_eps = config->lambda_min_eps;
    double last_z;
    double* last_lambda_l = new double[num_topics];
    do {
        last_z = z;
        memcpy(last_lambda_l, lambda_l, sizeof(double) * num_topics);
        get_descent_lambda_l(corpus, model, descent_lambda_l, index);

        bool has_neg_value_flag = false;
        for (int i = 0; !has_neg_value_flag && i < num_topics; i++) {
            lambda_l[i] += learn_rate * descent_lambda_l[i];
            if (lambda_l[i] < 0)has_neg_value_flag = true;
            //if (isnan(-doc->xi[i])) printf("doc->xi[i] nan\n");
        }
        if ( has_neg_value_flag || last_z > (z = get_lambda_l_function(corpus, model, index))) {
            learn_rate *= 0.1;
            z = last_z;
            eps = 10000;
            memcpy(lambda_l, last_lambda_l, sizeof(double) * num_topics);
        }
        else eps = util::norm2(last_lambda_l, lambda_l, num_topics);
        num_round ++;
    }
    while (num_round < max_lambda_iter && eps > lambda_min_eps);
    delete[] last_lambda_l;
    delete[] descent_lambda_l;
}

void update_ss(Document* doc, ctl_model* model, SS* ss) {
    pthread_mutex_lock(&mutex);
    int num_labels = model->num_labels;
    double* delta = doc->delta;
    double* u = doc->u;
    double* mu = model->mu;
    for (int i = 0; i < num_labels; ++ i) {
        ss->mu[i] += u[i];
        ss->sigma[i * num_labels + i] += delta[i];
    }
    for (int i = 0; i < num_labels; ++ i) {
        for (int j = 0; j < num_labels; ++ j) {
            ss->sigma[i * num_labels + j] += u[i] * u[j];
        }
    }
    pthread_mutex_unlock(&mutex);
}

void do_inference(Document* doc, ctl_model* model, Config* config, SS* ss) {
    int var_iter = 0;
    double lik_old = 0.0;
    double converged = 1;
    double lik, lik2;
    while ((converged > config->var_converence) && ((var_iter < config->max_var_iter || config->max_var_iter == -1))) {
        var_iter ++;
        inference_alpha_beta(doc, model);
        inference_delta(doc, model, config);
        inference_alpha_beta(doc, model);
        inference_u(doc, model, config);
        inference_alpha_beta(doc, model);
        inference_gamma(doc, model);
        lik = compute_doc_log_likehood(doc,model);
        converged = (lik_old -lik) / lik_old;
        lik_old = lik;
    }
    update_ss(doc, model, ss);
    return;
}

void* thread_inference(void* thread_data) {
    Thread_Data* thread_data_ptr = (Thread_Data*) thread_data;
    Document** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Config* config = thread_data_ptr->config;
    ctl_model* model = thread_data_ptr->model;
    SS* ss = thread_data_ptr->ss;
    for (int i = start; i < end; i++) {
        do_inference(corpus[i], model, config, ss);
    }
    return NULL;
}

void run_thread_inference(Document** corpus, ctl_model* model, Config* config, SS* ss) {
    int num_threads = config->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_docs = model->num_docs;
    int num_per_threads = num_docs/num_threads;
    int i;
    Thread_Data** thread_datas = new Thread_Data* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        thread_datas[i] = new Thread_Data(corpus, i * num_per_threads, (i+1)*num_per_threads, config, model, ss);
        pthread_create(&pthread_ts[i], NULL, thread_inference, (void*) thread_datas[i]);
    }
    thread_datas[i] = new Thread_Data(corpus, i * num_per_threads, num_docs, config, model, ss);
    pthread_create(&pthread_ts[i], NULL, thread_inference, (void*) thread_datas[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);
    for (i = 0; i < num_threads; i++) delete thread_datas[i];
    delete[] thread_datas;
}

bool delta_legal_size(const double &delta) {
    const double legal_size = 10.0;
    return fabs(delta) < legal_size;
}

void inference_delta(Document* doc, ctl_model* model, Config* config) {
    int doc_num_labels = doc->num_labels;
    for (int i = 0; i < doc_num_labels; ++ i) {
        //inference_delta_i(doc, model, config, i);
        inference_delta_i(doc, model, config, doc->labels_ptr[i]); 
    }
}

void inference_delta_i(Document* doc, ctl_model* model, Config* config, const int &index) {
    //printf("%d\n",index);
    double init_delta = util::random() + 1.0;
    double log_delta_i = log(init_delta);
    double df = 0.0, d2f = 0.0;
    int iter = 0;
    do {
        ++ iter;
        double delta_i = exp(log_delta_i);
        if (isinf(delta_i) || isinf(df) || isinf(d2f) || isnan(delta_i) || isnan(df) || isnan(d2f) || fabs(df) > NEWTON_MAX_DF) {
            init_delta = util::random() + 1.0;
            log_delta_i = log(init_delta);
            delta_i = init_delta;
        }
        df = df_delta_i(delta_i, index, doc, model);
        d2f = d2f_delta_i(delta_i, index, doc, model);
        log_delta_i -= (df * delta_i) / (d2f * delta_i * delta_i + df * delta_i);
    } while ((isnan(log_delta_i) || isnan(df) || fabs(df) > NEWTON_THRESH) && iter < NEWTON_EPOCH);
    if (delta_legal_size(exp(log_delta_i))) {
        doc->delta[index] = exp(log_delta_i);
    }
}

double get_u_function(Document* doc, ctl_model* model) {
    int* doc_labels_ptr = doc->labels_ptr;
    int* doc_words_cnt_ptr = doc->words_cnt_ptr;
    int num_labels = model->num_labels;
    int doc_num_labels = doc->num_labels;
    int doc_num_words = doc->num_words;
    int doc_num_all_words = doc->num_all_words;
    int num_topics = model->num_topics;
    double* lambda = model->lambda;
    double* mu = model->mu;
    double* doc_delta = doc->delta;
    double* log_gamma = doc->log_gamma;
    double alpha = doc->alpha;
    double beta = doc->beta;
    double res = 0.0;
    double* lambda_sum = new double[doc_num_labels];
    double* sigma = model->sigma;
    double* u = doc->u;

    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        lambda_sum[i] = 0.0;
        for (int j = 0; j < num_topics; ++ j) {
            lambda_sum[i] += lambda[label * num_topics + j];
        }
    }

    for (int n = 0; n < doc_num_words; ++ n) {
        for (int k = 0; k < num_topics; ++ k) {
            for (int i = 0; i < doc_num_labels; ++ i) {
                int label = doc_labels_ptr[i];
                res += doc_words_cnt_ptr[n] * exp(log_gamma[n * num_topics + k]) / alpha * exp(u[label] + doc_delta[label] * 0.5) * lambda[label * num_topics + k] / lambda_sum[i];
            }
        }
    }

    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        res -= 1.0 * doc_num_all_words / beta * exp(u[label] + doc_delta[label] * 0.5);
    }

    double* sum = new double[num_labels];
    double* sigma_inv = model->inv_sigma;
    for (int i = 0; i < num_labels; ++ i) {
        sum[i] = 0.0;
        for (int j = 0; j < num_labels; ++ j) {
            sum[i] += (u[j] - mu[j]) * sigma_inv[j * num_labels + i];
        }
        res -= 0.5 * sum[i] * (u[i] - mu[i]);
    }

    delete []sum;
    delete []lambda_sum;

    return res;
}

bool check_size(gsl_vector* gradient) {
    for (int i = 0; i < gradient->size; ++ i) {
        if (fabs(vget(gradient, i)) > PARAMS.gradient_size) return false;
    }
    return true;
}

void inference_u(Document* doc, ctl_model* model, Config* config) {
    gsl_multimin_function_fdf lambda_obj;
    const gsl_multimin_fdfminimizer_type* T;
    gsl_multimin_fdfminimizer* s;
    GslBundle bundle(doc, model);
    int iter = 0, i, j;
    int status;
    double f_old, converged;
    double* doc_u = doc->u;
    int doc_num_labels = doc->num_labels;
    int* doc_labels_ptr = doc->labels_ptr;
    gsl_vector* old_gradient;

    lambda_obj.f = &f_u;
    lambda_obj.df = &df_u;
    lambda_obj.fdf = &fdf_u;
    lambda_obj.n = doc_num_labels;
    lambda_obj.params = (void*)&bundle;

    // starting value
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc (T, doc_num_labels);

    gsl_vector* x = gsl_vector_calloc(doc_num_labels);
    for (i = 0; i < doc_num_labels; i++) vset(x, i, util::random());
    gsl_multimin_fdfminimizer_set(s, &lambda_obj, x, 1e-3, 1e-4);
    do {
        iter++;
        f_old = s->f;
        old_gradient = s->gradient;
        status = gsl_multimin_fdfminimizer_iterate(s);
        converged = fabs((f_old - s->f) / f_old);
        status = gsl_multimin_test_gradient(s->gradient, 1e-1);
        for (int i = 0; i < doc_num_labels; ++ i) {
            if (isnan(vget(s->x, i)) || isinf(vget(s->x, i))) vset(s->x, i, util::random());
        }
        if (!check_size(s->gradient)) break;
    } while ((status == GSL_CONTINUE) &&
            ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    if (iter == PARAMS.cg_max_iter) printf("warning: cg didn't converge (u) \n");

    for (i = 0; i < doc_num_labels; i++) {
        int label = doc_labels_ptr[i];
        doc_u[label] = vget(s->x, i);
    }

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}

double f_u(const gsl_vector* p, void* params) {
    GslBundle* bundle = (GslBundle*)params;
    Document* doc = bundle->doc;
    ctl_model* model = bundle->model;
    int* doc_labels_ptr = doc->labels_ptr;
    int* doc_words_cnt_ptr = doc->words_cnt_ptr;
    int num_labels = model->num_labels;
    int doc_num_labels = doc->num_labels;
    int doc_num_words = doc->num_words;
    int doc_num_all_words = doc->num_all_words;
    int num_topics = model->num_topics;
    double* lambda = model->lambda;
    double* mu = model->mu;
    double* doc_delta = doc->delta;
    double* log_gamma = doc->log_gamma;
    double alpha = doc->alpha;
    double beta = doc->beta;
    double res = 0.0;
    double* doc_u = new double[doc_num_labels];
    double* lambda_sum = new double[doc_num_labels];
    double* sigma = model->sigma;

    for (int i = 0; i < doc_num_labels; ++ i) doc_u[i] = vget(p, i);

    for (int i = 0; i < doc_num_labels; ++ i) {
        lambda_sum[i] = 0.0;
        int label = doc_labels_ptr[i];
        for (int j = 0; j < num_topics; ++ j) {
            lambda_sum[i] += lambda[label * num_topics + j];
        }
    }

    for (int n = 0; n < doc_num_words; ++ n) {
        for (int k = 0; k < num_topics; ++ k) {
            for (int i = 0; i < doc_num_labels; ++ i) {
                int label = doc_labels_ptr[i];
                res += doc_words_cnt_ptr[n] * exp(log_gamma[n * num_topics + k]) / alpha * exp(doc_u[i] + doc_delta[label] * 0.5) * lambda[label * num_topics + k] / lambda_sum[i];
            }
        }
    }

    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        res -= 1.0 * doc_num_all_words / beta * exp(doc_u[i] + doc_delta[label] * 0.5);
    }

    double* sum = new double[doc_num_labels];
    double* sigma_inv = model->inv_sigma;
    for (int i = 0; i < doc_num_labels; ++ i) {
        sum[i] = 0.0;
        int label_i = doc_labels_ptr[i];
        for (int j = 0; j < doc_num_labels; ++ j) {
            int label_j = doc_labels_ptr[j];
            sum[i] += (doc_u[j] - mu[label_j]) * sigma_inv[label_j * num_labels + label_i];
        }
        res -= 0.5 * sum[i] * (doc_u[i] - mu[label_i]);
    }

    delete []sum;
    delete []lambda_sum;
    delete []doc_u;

    return -res;
}

void df_u(const gsl_vector* p, void* params, gsl_vector* df) {
    GslBundle* bundle = (GslBundle*)params;
    Document* doc = bundle->doc;
    ctl_model* model = bundle->model;
    int* doc_labels_ptr = doc->labels_ptr;
    int* doc_words_cnt_ptr = doc->words_cnt_ptr;
    int num_labels = model->num_labels;
    int num_topics = model->num_topics;
    int doc_num_words = doc->num_words;
    int doc_num_labels = doc->num_labels;
    int doc_num_all_words = doc->num_all_words;
    double* doc_u = new double[doc_num_labels];
    double* doc_delta = doc->delta;
    double* sigma = model->sigma;
    double* mu = model->mu;
    double* doc_log_gamma = doc->log_gamma;
    double* lambda = model->lambda;
    double* lambda_sum = new double[doc_num_labels];
    double alpha = doc->alpha;
    double beta = doc->beta;

    double* res = new double[doc_num_labels];
    memset(res, 0, sizeof(double) * doc_num_labels);
    double* sigma_inv = model->inv_sigma;

    for (int i = 0; i < doc_num_labels; ++ i) {
        lambda_sum[i] = 0.0;
        int label = doc_labels_ptr[i];
        for (int j = 0; j < num_topics; ++ j) {
            lambda_sum[i] += lambda[label * num_topics + j];
        }
    }

    for (int i = 0; i < doc_num_labels; ++ i) {
        doc_u[i] = vget(p, i);
    }

    for (int i = 0; i < doc_num_labels; ++ i) {
        int label_i = doc_labels_ptr[i];
        res[i] = 0.0;
        for (int j = 0; j < doc_num_labels; ++ j) {
            int label_j = doc_labels_ptr[j];
            res[i] -= sigma_inv[label_j * num_labels + label_i] * (doc_u[j] - mu[label_j]);
        }
    }
    for (int n = 0; n < doc_num_words; ++ n) {
        for (int k = 0; k < num_topics; ++ k) {
            for (int i = 0; i < doc_num_labels; ++ i) {
                int label = doc_labels_ptr[i];
                res[i] += doc_words_cnt_ptr[n] * exp(doc_log_gamma[n * num_topics + k]) / alpha * exp(doc_u[i] + doc_delta[label] * 0.5) * lambda[label * num_topics + k] / lambda_sum[i];
            }
        }
    }
    for (int i = 0; i < doc_num_labels; ++ i) {
        int label = doc_labels_ptr[i];
        res[i] -= doc_num_all_words * 1.0 / beta * exp(doc_u[i] + doc_delta[label] * 0.5);
        vset(df, i, -res[i]);
    }

    delete []lambda_sum;
    delete []res;
    delete []doc_u;
}

void fdf_u(const gsl_vector* p, void* params, double* f, gsl_vector* df) {
    *f = f_u(p, params);
    df_u(p, params, df);
}
