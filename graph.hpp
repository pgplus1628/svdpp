#pragma once 
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>
#include <utility>
#include <unordered_map>
#include <sstream>

typedef uint32_t VidType;


template<typename ET>
struct Edge {
  VidType src;
  VidType dst;
  ET val;
  Edge() {}
  Edge(VidType _src, VidType _dst, ET _val) 
    : src(_src), dst(_dst), val(_val){}
};

template<typename ET>
class Graph { 
  public : 
  typedef struct Edge<ET> EdgeType;
  size_t strip_width;
  std::vector<EdgeType > edges;
  std::unordered_map<VidType, VidType> Uid2id;
  std::unordered_map<VidType, VidType> Vid2id;

  Graph(size_t strip_width_in) 
    : strip_width(strip_width_in)
  {
    edges.clear();
    Uid2id.clear();
    Vid2id.clear();
  }

  void load(std::string fname)
  {
    /* load graph */
    LOG(INFO) << " Loading graph " << fname;
    std::ifstream ifs(fname.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!ifs.good()) {
      LOG(FATAL) << "Error opning file " << fname << ".";
    }
    std::string line;
    std::unordered_map<VidType, VidType>::iterator it;
    while(std::getline(ifs, line) ) {
      if (line[0] == '#') {
        LOG(INFO) << line;
        continue;
      }
      VidType src, dst, tmp;
      ET val;
      std::stringstream ss(line);
      ss >> src >> dst >> val;
      it = Uid2id.find(src);
      if (it == Uid2id.end() ) {
        tmp = Uid2id.size();
        Uid2id[src] = tmp;
        src = tmp;
      } else { 
        src = it->second;
      }

      it = Vid2id.find(dst);
      if (it == Vid2id.end() ) {
        tmp = Vid2id.size();
        Vid2id[dst] = tmp;
        dst = tmp;
      } else { 
        dst = it->second;
      }
      
      edges.emplace_back(src, dst, val);

    }
    LOG(INFO) << " Load graph fininshed. Number of edges : " << edges.size() << ".";
    ifs.close();
    

    /* finalize graph */
    LOG(INFO) << " Finalize graph begin.";
    std::sort(edges.begin(), edges.end(), 
              [] (EdgeType const &A, EdgeType const &B) {
                return A.dst < B.dst;
              }); // sort by dst

    size_t ii = 0;

    bool ok = false;
    size_t cnt = 0;
    size_t p_dst = edges[0].dst;
    size_t beg = 0;
    for(size_t i = 0;i < edges.size();i ++ ) {
      if (p_dst != edges[i].dst) {
        cnt += 1;
        p_dst = edges[i].dst;
      } else { 
        continue;
      }
    
      if (cnt == strip_width) {
        std::sort(edges.begin() + beg, edges.begin() + i,
                  [] (EdgeType const &A, EdgeType const &B) {
                    if (A.src != B.src) return A.src < B.src;
                    return A.dst < B.dst;
                  });
        cnt = 0;
        beg = i;
      }
    }
  
    std::sort(edges.begin() + beg, edges.end(),
              [] (EdgeType const &A, EdgeType const &B) {
                if (A.src != B.src) return A.src < B.src;
                return A.dst < B.dst;
              });

    LOG(INFO) << " Finalize graph end.";
  }


  /* U side edge reduce */
  template<typename Ltype>
  void reduceU(std::vector<Ltype> &lvec, std::function<void(ET &eval, Ltype &l)> red_op) {
    for(auto &e: edges) {
      red_op(e.val, lvec[e.src]);
    }
  }

  /* accumulate edge values */
  template<typename T>
  void edge_apply(T& acc, std::function<void(ET &eval, T &t)> eapp) {
    for(auto &e : edges ) {
      eapp(e.val, acc);
    }
  }

  /* accumulate edge values */
  template<typename T1,
           typename T2,
           typename TACC>
  void edge_apply(std::vector<T1> &vec1, 
                  std::vector<T2> &vec2,
                  TACC &acc,
                  std::function<void(T1&, T2&, ET&, TACC&)> acc_op){
    for(auto &e : edges ) {
      acc_op(vec1[e.src], vec2[e.dst], e.val, acc);
    }
  }
                  


  /* do edge apply , apply to U and apply to V*/
  template<typename T>
  void edge_apply(std::vector<T> &vecU_out, std::vector<T> &vecV_out, 
                  std::function<void(T& u, ET &eval, T& v)> app_op) {
    for(auto &e : edges ) {
      app_op(vecU_out[e.src], e.val, vecV_out[e.dst]);
    }
  }


  /* do edge apply */
  template<typename F,
           typename W,
           typename L,
           typename R,
           typename S>
  void edge_apply(std::vector<F> &vecFU_out, std::vector<F> &vecFV_out,
                  std::vector<W> &vecWU_out, std::vector<W> &vecWV_out,
                  std::vector<L> &vecLU_out,
                  std::vector<R> &vecRU_out, std::vector<R> &vecRV_out,
                  std::vector<S> &vecSV_out,
                  std::function<void(F&, F&, W&, W&, L&, ET&, R&, R&, S&)> app_op) 
  {
    for(auto &e : edges ) {
      VidType s = e.src;
      VidType d = e.dst;
      app_op(vecFU_out[s], vecFV_out[d], 
             vecWU_out[s], vecWV_out[d],
             vecLU_out[s],
             e.val,
             vecRU_out[s], vecRV_out[d],
             vecSV_out[d]);
    }
  }

   /* <U, V> */
  std::pair<size_t, size_t> get_dim()
  {
    return std::make_pair(Uid2id.size(), Vid2id.size());
  }


  /* debug */
  void dump_id2id(std::string fname) 
  {
    std::string u_fname = fname + ".u.dat";
    std::ofstream ofs_u(u_fname.c_str(), std::ios_base::out | std::ios_base::binary);
    for(auto &kv : Uid2id) {
      ofs_u << kv.first << "\t" << kv.second << "\n";
    }
    ofs_u.close();

    std::string v_fname = fname + ".v.dat";
    std::ofstream ofs_v(v_fname.c_str(), std::ios_base::out | std::ios_base::binary);
    for(auto &kv : Vid2id) {
      ofs_v << kv.first << "\t" << kv.second << "\n";
    }
    ofs_v.close();
  }


};






















