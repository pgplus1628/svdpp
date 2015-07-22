#pragma once 
#include <vector>
#include <functional>
#include <algorithm>


template<typename VT>
void unary_app(std::vector<VT> &vec, std::function<void(VT&)> op)
{
  std::for_each(vec.begin(), vec.end(), op);
}

template<typename VT1, typename VT2>
void binary_app(std::vector<VT1> &vec1, std::vector<VT2> &vec2, std::function<void(VT1&, VT2&)> op)
{
  for(size_t i = 0;i < vec1.size(); i++) {
    op(vec1[i], vec2[i]);
  }
}


template<typename VT1, 
         typename VT2,
         typename VT3,
         typename VT4>
void quaternary_app(std::vector<VT1> &vec1, 
                std::vector<VT2> &vec2, 
                std::vector<VT2> &vec3, 
                std::vector<VT2> &vec4, 
                std::function<void(VT1&, VT2&)> op)
{
  for(size_t i = 0;i < vec1.size(); i++) {
    op(vec1[i], vec2[i], vec3[i], vec4[i]);
  }
}


