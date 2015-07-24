#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cmath>
#include <random>
#include <algorithm>

#include "graph.hpp"
#include "vec.hpp"

DEFINE_string(graph, "", "path to user item rating file. line format : <item> <user> <rating>, separated by tab.");
DEFINE_int32(max_iter, 1000000, "max iteration.");
DEFINE_int32(strip_width, 1024, "strip_width");

class RandGen {
  public :
  std::mt19937 gen;
  RandGen() {
    gen.seed(time(0));
  }
  double get_rand() {
    return std::uniform_real_distribution<> (-10.0, 10.0)(gen);
  }
};

static RandGen RG;


class SVDPP {
  public :
  static float itmBiasStep ;
  static float itmBiasReg  ;
  static float usrBiasStep ;
  static float usrBiasReg  ;
  static float usrFctrStep ;
  static float usrFctrReg  ;
  static float itmFctrStep ;
  static float itmFctrReg  ;
  static float itmFctr2Step;
  static float itmFctr2Reg ;
  static double MINVAL ;
  static double MAXVAL ;
  static double GLOBAL_MEAN ;
  static double STEP_DEC;

  static double rmse;


  static const size_t NLATENT = 128;

  /*
   * Edge type
   */
  struct Etype { 
    double obs;
    friend std::istream & operator >>(std::istream& is, struct Etype &eval){
      is >> eval.obs;
      return is;
    }
    friend std::ostream & operator <<(std::ostream& os, struct Etype &eval){
      os << eval.obs;
      return os;
    }
  };
  
  typedef struct Etype Etype;
  
  /* 
   * Feature vector type
   */
  struct Ftype { 
    double pvec[NLATENT];
    double bias;
    std::string to_string() {
      std::string ret;
      ret += "biase : "  + std::to_string(bias);
      ret += "\npvec : " ;
      std::for_each(pvec, pvec+NLATENT, [&](double &x) { ret += " " + std::to_string(x) ; });
      return ret;
    }
  };
  typedef struct Ftype Ftype;

  /* Weight type */
  struct Wtype { 
    double weight[NLATENT];
  };
  typedef struct Wtype Wtype;


  /* Label, sqrt(v.out_degree) */
  typedef double Ltype;

  struct Rtype {
    double delta_pvec[NLATENT];
    double delta_bias;
    std::string to_string() {
      std::string ret;
      ret += "biase : "  + std::to_string(delta_bias);
      ret += "\npvec : " ;
      std::for_each(delta_pvec, delta_pvec+NLATENT, [&](double &x) { ret += " " + std::to_string(x) ; });
      return ret;
    }
  };
  typedef struct Rtype Rtype;

  /* step type */
  struct Stype { 
    double step[NLATENT];
  };
  typedef struct Stype Stype;



  // ----------- / 
  // Functions 
  // ----------- /

  static void reset_l(Ltype &l) {
    l = 0.0;
  }

  static void map_l(Etype &e,Ltype &l) {
    l += 1.0;
  }

  static void update_l(Ltype &l) {
    l = 1.0 / std::sqrt(l);
  }

  static void reset_f(Ftype &f) {
    for(size_t i = 0;i < NLATENT;i ++) {
      f.pvec[i] = 0.0;
    }
    f.bias = 0.0;
  }

  static void rand_f(Ftype &f) {
    for(size_t i = 0;i < NLATENT;i ++) {
      f.pvec[i] = RG.get_rand();
    }
    f.bias = RG.get_rand();
  }
  

  static void reset_r(Rtype &r) {
    for(size_t i = 0;i < NLATENT;i ++) {
        r.delta_pvec[i] = 0.0;
        r.delta_bias = 0.0;
    }
  }

  static void reset_s(Stype &s) {
    for(size_t i = 0;i < NLATENT;i ++) {
      s.step[i] = 0.0;
    }
  }

  /* global mean edge_apply */
  static void gb_eapp(Etype &e, double &acc) {
    acc += e.obs;
  }


  static void reset_w(Wtype &w) {
    for(size_t i = 0;i < NLATENT;i ++ ) {
      w.weight[i] = 0.0;
    }
  }

  static void gather_weight(Wtype &w_user, Etype &e, Wtype &w_item) {
    for(size_t i = 0;i < NLATENT;i ++) {
      w_user.weight[i] += w_item.weight[i];
    }
  }


  static void gen_gradient(Ftype &f_user, Ftype &f_item, 
                         Wtype &w_user, Wtype &w_item,
                         Ltype &l_user,
                         Etype &e,
                         Rtype &r_user, Rtype &r_item, 
                         Stype &s_item)
  {
    double pred = GLOBAL_MEAN + f_user.bias + f_item.bias;
    for(size_t i = 0;i < NLATENT; i ++) {
      pred += f_user.pvec[i] * ( f_item.pvec[i] + w_item.weight[i]);
    }

    pred = std::min(pred, MAXVAL);
    pred = std::max(pred, MINVAL);

    float err = e.obs - pred;
    
    /* gen reduces */
    r_user.delta_bias += usrBiasStep*(err - usrBiasReg * f_user.bias);
    r_item.delta_bias += itmBiasStep*(err - itmBiasReg * f_item.bias);

    for(size_t i = 0;i < NLATENT;i ++) {
      r_user.delta_pvec[i] += usrFctrStep * (err * 
                  (f_item.pvec[i] - usrFctrReg * f_user.pvec[i]) );
    }

    for(size_t i = 0;i < NLATENT;i ++) {
      r_item.delta_pvec[i] += itmFctrStep * (err * 
                  (f_user.pvec[i] + w_user.weight[i]) - itmFctrReg * f_item.pvec[i]) ;
    }

    /* gen step */
    double _a = err  * itmFctr2Step * l_user ;
    double _b = itmFctr2Step * itmFctr2Reg;
    for(size_t i = 0; i < NLATENT;i ++) {
      s_item.step[i] += f_item.pvec[i] * _a - _b * w_item.weight[i];
    }
  }


  static void update_user(Rtype &r_user, Ftype &f_user){
    for(size_t i = 0;i < NLATENT;i ++) {
      f_user.pvec[i] += r_user.delta_pvec[i];
    }
    f_user.bias += r_user.delta_bias;
  }

  static void update_item(Rtype &r_item, Stype &s_item, Ftype &f_item, Wtype &w_item) {
    for(size_t i = 0;i < NLATENT;i ++) {
      f_item.pvec[i] += r_item.delta_pvec[i];
    }
    f_item.bias += r_item.delta_bias;
    for(size_t i = 0;i < NLATENT;i ++) {
      w_item.weight[i] += s_item.step[i];
    }
  }


  static void acc_error(Ftype &f_user, Ftype &f_item, Etype &e, double &rmse) {
    double pred = GLOBAL_MEAN + f_user.bias + f_item.bias;
    for(size_t i = 0;i < NLATENT;i ++) {
      pred += f_user.pvec[i] * f_item.pvec[i];
    }
    
    pred = std::min(MAXVAL, pred);
    pred = std::max(MINVAL, pred); 
    double err = (e.obs - pred) * (e.obs - pred);
    rmse += err;
  }

  static void update_k() {
    usrBiasStep *= STEP_DEC;
    itmBiasStep *= STEP_DEC;
    usrFctrStep *= STEP_DEC;
    itmFctrStep *= STEP_DEC;
    itmFctr2Step *= STEP_DEC;
  }
  

};


float SVDPP::itmBiasStep  = 1e-9;
float SVDPP::itmBiasReg   = 1e-9;
float SVDPP::usrBiasStep  = 1e-9;
float SVDPP::usrBiasReg   = 1e-9;
float SVDPP::usrFctrStep  = 1e-9;
float SVDPP::usrFctrReg   = 1e-9;
float SVDPP::itmFctrStep  = 1e-9;
float SVDPP::itmFctrReg   = 1e-9;
float SVDPP::itmFctr2Step = 1e-9;
float SVDPP::itmFctr2Reg  = 1e-9;
double SVDPP::MINVAL = -1e+100;
double SVDPP::MAXVAL = 1e+100;
double SVDPP::GLOBAL_MEAN = 0.0;
double SVDPP::STEP_DEC = 0.99;

double SVDPP::rmse = 0.0;



int main(int argc, char ** argv) {

  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  Graph<SVDPP::Etype> * graph = new Graph<SVDPP::Etype>(FLAGS_strip_width);
  /* load graph */
  graph->load(FLAGS_graph);

  LOG(INFO) << " Graph Load Finished.";

  size_t u_len = graph->get_dim().first;
  size_t v_len = graph->get_dim().second;
  size_t num_edges = graph->get_num_edges();

  //graph->dump_id2id(FLAGS_graph);

  LOG(INFO) << " graph dim : " << u_len << " , " << v_len;

  std::vector<SVDPP::Ftype> *f_user = new std::vector<SVDPP::Ftype>(u_len);
  std::vector<SVDPP::Ftype> *f_item = new std::vector<SVDPP::Ftype>(v_len);

  std::vector<SVDPP::Wtype> *w_user = new std::vector<SVDPP::Wtype>(u_len);
  std::vector<SVDPP::Wtype> *w_item = new std::vector<SVDPP::Wtype>(v_len);

  std::vector<SVDPP::Rtype> *r_user = new std::vector<SVDPP::Rtype>(u_len);
  std::vector<SVDPP::Rtype> *r_item = new std::vector<SVDPP::Rtype>(v_len);

  std::vector<SVDPP::Ltype> *l_user = new std::vector<SVDPP::Ltype>(u_len);
  std::vector<SVDPP::Stype> *s_item = new std::vector<SVDPP::Stype>(v_len);

  /* init */
  unary_app<SVDPP::Ltype>(*l_user, SVDPP::reset_l);
  graph->reduceU<SVDPP::Ltype>(*l_user, SVDPP::map_l);
  unary_app<SVDPP::Ltype>(*l_user, SVDPP::update_l);


  // GLOBAL_MEAN
  graph->edge_apply<double>(SVDPP::GLOBAL_MEAN, SVDPP::gb_eapp);


  /* train */
  //unary_app<SVDPP::Ftype>(*f_user, SVDPP::reset_f);
  //unary_app<SVDPP::Ftype>(*f_item, SVDPP::reset_f);
  unary_app<SVDPP::Ftype>(*f_user, SVDPP::rand_f);
  unary_app<SVDPP::Ftype>(*f_item, SVDPP::rand_f);


  for(size_t it = 0; it < FLAGS_max_iter; it ++) {
    /* reset r_user r_item s_item */
    unary_app<SVDPP::Rtype>(*r_user, SVDPP::reset_r);
    unary_app<SVDPP::Rtype>(*r_item, SVDPP::reset_r);
    unary_app<SVDPP::Stype>(*s_item, SVDPP::reset_s);
    unary_app<SVDPP::Wtype>(*w_user, SVDPP::reset_w);


    /* user gather weights */
    graph->edge_apply<SVDPP::Wtype>(*w_user, *w_item, SVDPP::gather_weight);

    /* edge apply */
    graph->edge_apply<SVDPP::Ftype, SVDPP::Wtype, 
                      SVDPP::Ltype, SVDPP::Rtype, 
                      SVDPP::Stype>(*f_user,
                                    *f_item, 
                                    *w_user,
                                    *w_item,
                                    *l_user,
                                    *r_user,
                                    *r_item,
                                    *s_item,
                                    SVDPP::gen_gradient);


    if (it % 100 == 0) {
      dump_vec<SVDPP::Rtype>(*r_user, "r_user_" + std::to_string(it) + ".dat");
    }

    /* update f_user and f_item */
    binary_app<SVDPP::Rtype, SVDPP::Ftype>(*r_user, *f_user, SVDPP::update_user);
    quaternary_app<SVDPP::Rtype, SVDPP::Stype, 
                   SVDPP::Ftype, SVDPP::Wtype>(*r_item, *s_item, 
                                               *f_item, *w_item,
                                               SVDPP::update_item);

    SVDPP::rmse = 0.0;
    /* accumulate rmse */
    graph->edge_apply<SVDPP::Ftype, SVDPP::Ftype, double>
                     (*f_user, *f_item, SVDPP::rmse, SVDPP::acc_error);
    double r = SVDPP::rmse;
    r = std::sqrt( r / double(num_edges) );
    SVDPP::rmse = r;

    /* update k */
    if (it % 60 == 0)
      SVDPP::update_k();

/*
    if (it % 100 == 0) {
      LOG(INFO) << " ----------- " ;
      dump_vec<SVDPP::Ftype>(*f_user, "f_user_" + std::to_string(it) + ".dat");
    }
*/

    LOG(INFO) << " SVDPP::iteration " << it <<  " rmse : " << SVDPP::rmse << " end.";
  }

  return 0;
}


