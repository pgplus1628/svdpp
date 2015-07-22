#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cmath>

#include "graph.hpp"
#include "vec.hpp"

DEFINE_string(graph, "", "path to user item rating file. line format : <item> <user> <rating>, separated by tab.");
DEFINE_int32(max_iter, 1000000, "max iteration.");


class SVDPP {
  static float itmBiasStep  = 1e-4;
  static float itmBiasReg   = 1e-4;
  static float usrBiasStep  = 1e-4;
  static float usrBiasReg   = 1e-4;
  static float usrFctrStep  = 1e-4;
  static float usrFctrReg   = 1e-4;
  static float itmFctrStep  = 1e-4;
  static float itmFctrReg   = 1e-4;
  static float itmFctr2Step = 1e-4;
  static float itmFctr2Reg  = 1e-4;


  static size_t NLATENT;
  static double MINVAL = -1e+100;
  static double MAXVAL = 1e+100;

  static double GLOBAL_MEAN = 0.0;

  /*
   * Edge type
   */
  struct Etype { 
    double obs;
  };
  
  typedef struct Etype Etype;
  
  /* 
   * Feature vector type
   */
  struct Ftype { 
    double pvec[NLATENT];
    double bias;
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
      f.bias = 0.0;
    }
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
    acc += e;
  }


  static void reset_weight(Wtype &w) {
    for(size_t i = 0;i < NLATENT;i ++ ) {
      w.weight[i] = 0.0;
    }
  }

  static void gather_weight(Wtype &w_user, Etype &e, Wtype &w_item) {
    for(size_t i = 0;i < NLATENT;i ++) {
      w_user.weight[i] += w_item.weight[i];
    }
  }


  static void triplet_apply(Ftype &f_user, Ftype &f_item, 
                            Wtype &w_user, Wtype &w_item,
                            Ltype &l_user,
                            Etype &e,
                            Rtype &r_user, Rtype &r_item, Stype &s_item)
  {
    double pred = GLOBAL_MEAN + f_user.bias + f_item.bias;
    for(size_t i = 0;i < NLATENT; i ++) {
      pred += f_user.pvec[i] * ( f_item.pvec[i] + w_item.weight[i]);
    }

    pred = std::min(pred, MAXVAL);
    pred = std::max(pred, MINVAL);

    float err = e.obs - pred;
    
    /* gen reduces */
    r_user.delta_bias = usrBiasStep*(err - usrBiasReg * f_user.bias);
    r_item.delta_bias = itmBiasStep*(err - itmBiasReg * f_item.bias);

    for(size_t i = 0;i < NLATENT;i ++) {
      r_user.delta_pvec[i] = usrFctrStep * (err * 
                  (f_item.pvec[i] - usrFctrReg * f_user.pvec[i]) );
    }

    for(size_t i = 0;i < NLATENT;i ++) {
      r_item.delta_pvec[i] = itmFctrStep * (err * 
                  (f_user.pvec[i] + f_user.weight[i] - itmFctrReg * f_item.pvec[i]) );
    }

    /* gen step */
    double _a = err * itmFctr2Step * l_user;
    double _b = itmFctr2Step * itmFctr2Reg;
    for(size_t i = 0; i < NLATENT;i ++) {
      s_item.step[i] = f_item.pvec[i] * _a - _b * f_item.weight[i];
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
      f_item.pvec[i] += r_item.pvec[i];
    }
    f_item.bias += r_item.bias;
    for(size_t i = 0;i < NLATENT;i ++) {
      w_item.weight[i] += s_item.step[i];
    }
  }

};





int main(int argc, char ** argv) {

  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  Graph<Etype> * graph = new Graph<Etype>();
  size_t u_len = graph->get_dim().first();
  size_t v_len = graph->get_dim().second();

  std::vector<Ftype> *f_user = new std::vector<Ftype>(u_len);
  std::vector<Ftype> *f_item = new std::vector<Ftype>(v_len);

  std::vector<Wtype> *w_user = new std::vector<Wtype>(u_len);
  std::vector<Wtype> *w_item = new std::vector<Wtype>(v_len);

  std::vector<Ltype> *l_user = new std::vector<Ltype>(u_len);
  std::vector<Stype> *s_item = new std::vector<Stype>(v_len);

  /* load graph */
  graph->load(FLAGS_graph);

  /* init */
  unary_app<Ltype>(*l_user, SVDPP::reset_l);
  graph->reduce<Ltype>(SVDPP::map_l, *luser);
  unary_app<Ltype>(*l_user, SVDPP::update_l);

  // GLOBAL_MEAN
  graph->edge_apply<double>(SVDPP::GLOBAL_MEAN, SVDPP::gb_eapp);

  /* train */
  for(size_t it = 0; it < FLAGS_max_iter; it ++) {
    LOG(INFO) << " SVDPP::iteration " << it << " begin."
    /* reset r_user r_item s_item */
    unary_app<Rtype>(*r_user, reset_r);
    unary_app<Rtype>(*r_item, reset_r);
    unary_app<Stype>(*s_item, reset_s);


    /* user gather weights */
    graph->edge_apply<Wtype>(*w_user, *w_item, SVDPP::gather_weight);

    /* edge apply */
    graph->edge_apply<Ftype, Wtype, Ltype, Rtype, Stype>(*f_user,
                                                         *f_item, 
                                                         *w_user,
                                                         *w_item,
                                                         *l_user,
                                                         *r_user,
                                                         *r_item,
                                                         *s_item);

    /* update f_user and f_item */
    binary_app<Rtype, Ftype>(*r_user, *f_user, SVDPP::update_user);
    quaternary_app<Rtype, Stype, Ftpye, Wtype>(*r_item, *s_item, 
                                               *f_item, *w_item,
                                               SVDPP::update_user);


    LOG(INFO) << " SVDPP::iteration " << it << " end."
  }






  return 0;
}


