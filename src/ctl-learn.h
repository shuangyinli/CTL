/*
 * ctl-learn.h
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#ifndef CTL_LEARN_H
#define CTL_LEARN_H

#include "utils.h"
#include "ctl.h"
#include "ctl-ss.h"

void learn_phi(Document** corpus, ctl_model* model);
void learn_mu(ctl_model* model, SS* ss);
void learn_sigma(ctl_model* model, SS* ss);

#endif
