/*
 * ctl-estimate.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#ifndef CTL_ESTIMATE_H
#define CTL_ESTIMATE_H

#include "utils.h"
#include "ctl.h"

double compute_doc_log_likehood(Document* doc, ctl_model* model);
double compute_doc_log_likehood2(Document* doc, ctl_model* model);
double compute_corpus_log_likehood(Document** corpus, ctl_model* model);
double compute_corpus_log_likehood2(Document** corpus, ctl_model* model);

#endif
