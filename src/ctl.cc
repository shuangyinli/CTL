/*
 * ctl.cc
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */
#include "ctl.h"
#include "ctl-inference.h"
#include "ctl-estimate.h"
#include "ctl-learn.h"
#include "ctl-ss.h"
#include "utils.h"
#include "params.h"

#include <assert.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

bool Config::print_debuginfo = true;

void Document::init_delta_u(const int &num_all_labels_) {
  num_all_labels = num_all_labels_;
  delta = new double[num_all_labels];
  u = new double[num_all_labels];
  memset(delta, 0, sizeof(double) * num_all_labels);
  for (int i = 0; i < num_all_labels; ++ i) u[i] = 0;
}

Document** read_data(char* filename,int num_topics,int& num_words, int& num_docs, int& num_labels, int& num_all_words) {
  num_words = 0;
  num_docs = 0;
  num_labels = 0;
  num_all_words = 0;
  FILE* fp = fopen(filename,"r"); //calcaulte the file line num
  char c;
  while((c=getc(fp))!=EOF) {
    if (c=='\n') num_docs++;
  }
  fclose(fp);
  fp = fopen(filename,"r");
  int doc_num_labels;
  int doc_num_words;
  char str[10];
  Document** corpus = new Document* [num_docs + 10];
  num_docs = 0;
  while(fscanf(fp,"%d",&doc_num_labels) != EOF) {
    int* labels_ptr = new int[doc_num_labels];
    for (int i = 0; i < doc_num_labels; i++) {
      fscanf(fp,"%d",&labels_ptr[i]);
      num_labels = num_labels > labels_ptr[i]?num_labels:labels_ptr[i];
    }
    fscanf(fp,"%s",str); //read @
    fscanf(fp,"%d", &doc_num_words);
    int* words_ptr = new int[doc_num_words];
    int* words_cnt_ptr = new int [doc_num_words];
    for (int i =0; i < doc_num_words; i++) {
      fscanf(fp,"%d:%d", &words_ptr[i],&words_cnt_ptr[i]);
      num_words = num_words < words_ptr[i]?words_ptr[i]:num_words;
      num_all_words += words_cnt_ptr[i];
    }
    corpus[num_docs++] = new Document(labels_ptr, words_ptr, words_cnt_ptr, doc_num_labels, doc_num_words, num_topics);
  }
  fclose(fp);
  num_words ++;
  num_labels ++;

  for (int i = 0; i < num_docs; ++ i) corpus[i]->init_delta_u(num_labels);
  printf("num_docs: %d\nnum_labels: %d\nnum_words:%d\n",num_docs,num_labels,num_words);
  return corpus;
}

void Config::read_settingfile(char* settingfile) {
  FILE* fp = fopen(settingfile,"r");
  char key[100];
  while (fscanf(fp,"%s",key)!=EOF){
    if (strcmp(key,"pi_learn_rate")==0) {
      fscanf(fp,"%lf",&pi_learn_rate);
      continue;
    }
    if (strcmp(key,"max_pi_iter") == 0) {
      fscanf(fp,"%d",&max_pi_iter);
      continue;
    }
    if (strcmp(key,"pi_min_eps") == 0) {
      fscanf(fp,"%lf",&pi_min_eps);
      continue;
    }
    if (strcmp(key,"xi_learn_rate") == 0) {
      fscanf(fp,"%lf",&xi_learn_rate);
      continue;
    }
    if (strcmp(key,"max_xi_iter") == 0) {
      fscanf(fp,"%d",&max_xi_iter);
      continue;
    }
    if (strcmp(key,"xi_min_eps") == 0) {
      fscanf(fp,"%lf",&xi_min_eps);
      continue;
    }
    if (strcmp(key,"max_em_iter") == 0) {
      fscanf(fp,"%d",&max_em_iter);
      continue;
    }
    if (strcmp(key,"num_threads") == 0) {
      fscanf(fp, "%d", &num_threads);
    }
    if (strcmp(key, "var_converence") == 0) {
      fscanf(fp, "%lf", &var_converence);
    }
    if (strcmp(key, "max_var_iter") == 0) {
      fscanf(fp, "%d", &max_var_iter);
    }
    if (strcmp(key, "em_converence") == 0) {
      fscanf(fp, "%lf", &em_converence);
    }
    if (strcmp(key, "level") == 0) {
      fscanf(fp, "%d", &level);
    }
  }
}

void Document::init() {
  num_all_words = 0;
  for (int i = 0; i < num_words; i++) {
    num_all_words += words_cnt_ptr[i];
    double sum_log_gamma = 0.0;
    for (int k = 0; k < num_topics; k++) {
      log_gamma[i * num_topics + k] = util::random();
      sum_log_gamma += log_gamma[i * num_topics + k]; 
    }
    for (int k = 0; k < num_topics; k++) {
      log_gamma[i * num_topics + k] /= sum_log_gamma;
      log_gamma[i * num_topics + k] = log(log_gamma[i * num_topics + k]);
    }
  }
}
void print_mat(double* mat, int row, int col, char* filename) {
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      fprintf(fp,"%lf ",mat[i*col + j]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void print_documents_u(Document** corpus, ctl_model* model, char* output_dir) {
  char filename[1000];
  int num_docs = model->num_docs;
  sprintf(filename, "%s/doc-u.txt", output_dir);
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < num_docs; ++ i) {
    Document* doc = corpus[i];
    for (int k = 0; k < model->num_labels; ++ k) fprintf(fp, "%lf ", doc->u[k]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void print_documents_topics(Document** corpus, int num_docs, char* output_dir) {
  char filename[1000];
  sprintf(filename, "%s/doc-topics-dis.txt", output_dir);
  char liks_file[1000];
  sprintf(liks_file, "%s/likehoods.txt", output_dir);
  FILE* liks_fp = fopen(liks_file, "w");
  FILE* fp = fopen(filename,"w");
  for (int i = 0; i < num_docs; i++) {
    Document* doc = corpus[i];
    fprintf(fp, "%lf", doc->topic[0]);
    fprintf(liks_fp, "%lf\n", doc->lik);
    for (int k = 1; k < doc->num_topics; k++) fprintf(fp, " %lf", doc->topic[k]);
    fprintf(fp, "\n");
  }
  fclose(fp);
  fclose(liks_fp);
}

void print_para(Document** corpus, int num_round, char* model_root, ctl_model* model) {
  for (int i = 0; i < model->num_labels; ++ i) {
    for (int j = 0; j < model->num_labels; ++ j) {
      assert(model->sigma[i * model->num_labels + j] - model->sigma[j * model->num_labels + i] < 1);
    }
  }

  char sigma_file[1000];
  char inv_sigma_file[1000];
  char mu_file[1000];
  char Lambda_file[1000];
  char phi_file[1000];
  char lambda_file[1000];
  char topic_dis_file[1000];
  char u_file[1000];

  if (num_round != -1) {
    sprintf(sigma_file, "%s/%03d.sigma", model_root, num_round);
    sprintf(inv_sigma_file, "%s/%03d.inv_sigma", model_root, num_round);
    sprintf(mu_file, "%s/%03d.mu", model_root, num_round);
    sprintf(Lambda_file, "%s/%03d.Lambda", model_root, num_round);
    sprintf(phi_file, "%s/%03d.phi", model_root, num_round);
    sprintf(lambda_file,"%s/%03d.lambda", model_root, num_round);
    sprintf(topic_dis_file,"%s/%03d.topic_dis", model_root, num_round);
    sprintf(u_file,"%s/%03d.u", model_root, num_round);
  }
  else {
    sprintf(sigma_file, "%s/final.sigma", model_root);
    sprintf(inv_sigma_file, "%s/final.inv_sigma", model_root);
    sprintf(mu_file, "%s/final.mu", model_root);
    sprintf(Lambda_file, "%s/final.Lambda", model_root);
    sprintf(phi_file, "%s/final.phi", model_root);
    sprintf(lambda_file,"%s/final.lambda", model_root);
    sprintf(topic_dis_file,"%s/final.topic_dis", model_root);
    sprintf(u_file,"%s/final.u", model_root);
  }

  print_mat(model->sigma, model->num_labels, model->num_labels, sigma_file);
  print_mat(model->inv_sigma, model->num_labels, model->num_labels, inv_sigma_file);
  print_mat(model->mu, 1, model->num_labels, mu_file);
  print_mat(model->Lambda, 1, model->num_topics, Lambda_file);
  print_mat(model->log_phi, model->num_topics, model->num_words, phi_file);
  print_mat(model->lambda, model->num_labels, model->num_topics, lambda_file);

  FILE* topic_dis_fp = fopen(topic_dis_file,"w");
  FILE* u_fp = fopen(u_file, "w");
  int num_docs = model->num_docs;
  for (int d = 0; d < num_docs; d++) {
    Document* doc = corpus[d];
    for (int k = 0; k < doc->num_topics; k++)fprintf(topic_dis_fp, "%lf ", doc->topic[k]);
    for (int k = 0; k < model->num_labels; ++ k) fprintf(u_fp, "%lf ", doc->u[k]);
    fprintf(topic_dis_fp, "\n");
    fprintf(u_fp, "\n");
  }
  fclose(u_fp);
  fclose(topic_dis_fp);
}

void print_lik(double* likehood_record, int num_round, char* model_root) {
  char lik_file[1000];
  sprintf(lik_file, "%s/likehood.dat", model_root);
  FILE* fp = fopen(lik_file,"w");
  for (int i = 0; i <= num_round; i++) {
    fprintf(fp, "%03d %lf\n", i, likehood_record[i]);
  }
  fclose(fp);
}

void print_model_info(char* model_root, int num_words, int num_labels,int num_topics) {
  char filename[1000];
  sprintf(filename, "%s/model.info",model_root);
  FILE* fp = fopen(filename,"w");
  fprintf(fp, "num_labels: %d\n", num_labels);
  fprintf(fp, "num_words: %d\n", num_words);
  fprintf(fp, "num_topics: %d\n", num_topics);
  fclose(fp);
}

void print_gamma_sum(char* model_root, Document** corpus, int num_docs, int num_topics) {
  char filename[1000];
  sprintf(filename, "%s/gamma_sum.dat",model_root);
  FILE* fp = fopen(filename, "w");
  double* sum = new double[num_topics];
  for (int d = 0; d < num_docs; ++ d) {
    memset(sum, 0, sizeof(double) * num_topics);
    int* words_ptr = corpus[d]->words_ptr;
    int* words_cnt_ptr = corpus[d]->words_cnt_ptr;
    int num_words = corpus[d]->num_words;
    double* log_gamma = corpus[d]->log_gamma;
    for (int i = 0; i < num_words; ++ i) {
      for (int j = 0; j < num_topics; ++ j) {
        sum[j] += words_cnt_ptr[i] * exp(log_gamma[i * num_topics + j]);
      }
    }
    for (int i = 0; i < num_topics; ++ i) fprintf(fp, "%lf ", sum[i]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void ctl_model::set_model(ctl_model* model) {
  memcpy(sigma, model->sigma, sizeof(double) * num_labels * num_labels);
  memcpy(inv_sigma, model->inv_sigma, sizeof(double) * num_labels * num_labels);
  memcpy(mu, model->mu, sizeof(double) * num_labels);
  memcpy(Lambda, model->Lambda, sizeof(double) * num_topics);
  memcpy(lambda, model->lambda, sizeof(double) * num_labels * num_topics);
  memcpy(log_phi, model->log_phi, sizeof(double) * num_topics * num_words);
}

void ctl_model::read_model_info(char* model_root) {
  char filename[1000];
  sprintf(filename, "%s/model.info",model_root);
  printf("%s\n",filename);
  FILE* fp = fopen(filename,"r");
  char str[100];
  int value;
  while (fscanf(fp,"%s%d",str,&value)!=EOF) {
    if (strcmp(str,"num_labels:") == 0)num_labels = value;
    if (strcmp(str, "num_words:") == 0)num_words = value;
    if (strcmp(str, "num_topics:") == 0)num_topics = value;
  }
  printf("num_labels: %d\nnum_words: %d\nnum_topics: %d\n",num_labels,num_words, num_topics);
  fclose(fp);
}

double* ctl_model::load_mat(char* filename, int row, int col) {
  FILE* fp = fopen(filename,"r");
  double* mat = new double[row * col];
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      fscanf(fp, "%lf", &mat[i*col+j]);
    }
  }
  fclose(fp);
  return mat;
}

void begin_ctl(char* inputfile, char* settingfile,int num_topics, char* model_root) {
  setbuf(stdout,NULL);
  int num_docs;
  int num_words;
  int num_labels;
  int num_all_words;
  srand(unsigned(time(0)));
  Document** corpus = read_data(inputfile,num_topics,num_words,num_docs,num_labels, num_all_words);
  puts("Read Data Finish!!");
  default_params();

  ctl_model* model = new ctl_model(num_docs,num_words,num_topics,num_labels,num_all_words);
  ctl_model* old_model = new ctl_model(num_docs,num_words,num_topics,num_labels,num_all_words);
  SS* ss = new SS(num_labels * num_labels, num_labels);
  Config config = Config(settingfile);
  print_model_info(model_root, num_words, num_labels, num_topics);

  char time_log_filename[100];
  sprintf(time_log_filename, "%s/costtime.log", model_root); 
  FILE* time_log_fp = fopen(time_log_filename,"w");
  setbuf(time_log_fp, NULL);

  time_t learn_begin_time = time(0);
  int num_round = 0;
  printf("cal likehood...\n");
  double lik = compute_corpus_log_likehood(corpus, model); 
  double lik2 = compute_corpus_log_likehood2(corpus, model);
  double old_lik;
  printf("lik %lf\n", lik);
  double plik;
  double* likehood_record = new double [config.max_em_iter];
  likehood_record[0] = lik;
  double converged = 1;

  //Initialize
  init_lambda(model->lambda, model->num_labels * model->num_topics);
  puts("Init Lambda Finish!!");
  inference_Lambda(model->Lambda, model->lambda, model->num_topics, model->num_labels, 0);
  init_sigma_mu_phi(model);
  //Finish Initialize
  //
  do {
    //Init SS
    old_model->set_model(model);
    old_lik = lik2;
    ss->init();
    time_t cur_round_begin_time = time(0);
    plik = lik;
    printf("Round %d begin...\n", num_round ++);
    printf("inference...\n");
    //E-step
    run_thread_inference(corpus, model, &config, ss);
    inference_lambda(corpus, model, &config);
    inference_Lambda(model->Lambda, model->lambda, model->num_topics, model->num_labels, 0);
    if (compute_corpus_log_likehood2(corpus, model) < old_lik) model->set_model(old_model);

    //M-step
    printf("learn phi...\n");
    learn_phi(corpus, model);
    printf("learn mu...\n");
    learn_mu(model, ss);
    printf("learn sigma...\n");
    learn_sigma(model, ss);
    printf("cal likehood...\n");
    lik = compute_corpus_log_likehood(corpus, model);
    lik2 = compute_corpus_log_likehood2(corpus, model);
    double perplex = exp(-lik2/model->num_all_words);
    converged = (plik - lik) / plik;
    unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;
    printf("Round %d: likehood=%lf likehood2=%lf perplex=%lf converged=%lf cost_time=%u secs.\n",num_round,lik,lik2,perplex,converged, cur_round_cost_time);
    likehood_record[num_round] = lik;
  } while (num_round < config.max_em_iter);
  unsigned int learn_cost_time = time(0) - learn_begin_time;
  if (time_log_fp) {
    fprintf(time_log_fp, "all learn runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
    fclose(time_log_fp);
  }
  print_lik(likehood_record, num_round, model_root);
  print_para(corpus, -1, model_root, model);

  for (int i = 0; i < num_docs; i++) delete corpus[i];
  delete[] likehood_record;
  delete[] corpus;
  delete model;
  delete ss;
}

void infer_ctl(char* test_file, char* settingfile, char* model_root,char* prefix,char* out_dir=NULL) {
  setbuf(stdout,NULL);
  int num_docs;
  int num_words;
  int num_labels;
  ctl_model* model = new ctl_model(model_root, prefix);
  int num_topics = model->num_topics;
  srand(unsigned(time(0)));
  Document** corpus = read_data(test_file,num_topics,num_words,num_docs,num_labels, model->num_all_words);
  model->num_docs = num_docs;
  Config config = Config(settingfile); 
  SS* ss = new SS(num_labels * num_labels, num_labels);

  run_thread_inference(corpus, model, &config, ss);
  double lik = compute_corpus_log_likehood(corpus, model);
  double lik2 = compute_corpus_log_likehood2(corpus, model);
  double perplex = exp(-lik/model->num_all_words);
  printf("likehood: %lf perplexity:%lf num all words: %d\n", lik, perplex,model->num_all_words);
  if (out_dir) {
    print_documents_topics(corpus, model->num_docs, out_dir);
    print_documents_u(corpus, model, out_dir);
  }

  for (int i = 0; i < num_docs; i++) {
    delete corpus[i];
  }
  delete[] corpus;
  delete model;
  delete ss;
}

void get_gamma(char* test_file, char* settingfile, char* model_root,char* prefix,char* out_dir = NULL) {
  setbuf(stdout,NULL);
  int num_docs;
  int num_words;
  int num_labels;
  ctl_model* model = new ctl_model(model_root, prefix);
  int num_topics = model->num_topics;
  srand(unsigned(time(0)));
  Document** corpus = read_data(test_file,num_topics,num_words,num_docs,num_labels, model->num_all_words);
  model->num_docs = num_docs;
  Config config = Config(settingfile); 
  SS* ss = new SS(num_labels * num_labels, num_labels);
  run_thread_inference(corpus, model, &config, ss);
  print_gamma_sum(out_dir, corpus, num_docs, model->num_topics);
  delete ss;
  delete model;
}

int main(int argc, char* argv[]) {
  if (argc <= 1 || (!(strcmp(argv[1],"est") == 0 && argc == 6)  && !(strcmp(argv[1],"inf") == 0 && (argc == 6||argc==7)) && ! (strcmp(argv[1], "lda-inf") == 0 && argc == 7) && !(strcmp(argv[1], "gamma") == 0 && argc == 7))) {
    printf("usage1: ./ctl est <input data file> <setting.txt> <num_topics> <model save dir>\n");
    printf("usage2: ./ctl inf <input data file> <setting.txt> <model dir> <prefix> <output dir>\n");
    printf("usage3: ./ctl lda-inf <input data file> <setting.txt> <model dir> <prefix> <output dir>\n");
    return 1;
  }
  if (argc > 1 && strcmp(argv[1],"est") == 0) begin_ctl(argv[2],argv[3],atoi(argv[4]),argv[5]);
  if (argc > 1 && strcmp(argv[1], "inf") == 0){
    if (argc==6) {
      infer_ctl(argv[2], argv[3],argv[4],argv[5]);
    }
    else {
      infer_ctl(argv[2],argv[3],argv[4],argv[5],argv[6]);
    }
  }
  if (argc > 1 && strcmp(argv[1], "gamma") == 0) {
    if (argc == 7) get_gamma(argv[2], argv[3], argv[4], argv[5], argv[6]);
  }
  return 0;
}

