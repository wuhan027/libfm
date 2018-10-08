// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// fm_model.h: Model for Factorization Machines
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.
//机器学习算法实现解析——libFM之libFM的模型处理部分 https://blog.csdn.net/google19890102/article/details/72866290

#ifndef FM_MODEL_H_
#define FM_MODEL_H_

#include "../util/matrix.h"
#include "../util/fmatrix.h"

#include "fm_data.h"


class fm_model {
 public:
  fm_model(); // 构造函数，主要完成参数的初始化
  void debug();
  void init();  // 初始化函数，主要用于生成各维度系数的初始值
    // 对样本进行预测
  double predict(sparse_row<FM_FLOAT>& x);
  double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);
  void saveModel(std::string model_file_path);
  int loadModel(std::string model_file_path);

    //fm模型中的参数
  double w0;  // 常数项
  DVectorDouble w;  // 一次项的系数
  DMatrixDouble v;  // 交叉项的系数矩阵

  // the following values should be set:
  uint num_attribute; // 特征的个数

  bool k0, k1;  // 是否包含常数项和一次项
  int num_factor; // 交叉项因子的个数

  double reg0;  // 常数项的正则参数
  double regw, regv;  // 一次项和交叉项的正则系数

  double init_stdev;  // 初始化参数时的方差
  double init_mean; // 初始化参数时的均值

 private:
  void splitString(const std::string& s, char c, std::vector<std::string>& v);

  DVector<double> m_sum, m_sum_sqr; // 分别对应着交叉项的中的两项
};

// Implementation
fm_model::fm_model() {
  num_factor = 0;
  init_mean = 0;
  init_stdev = 0.01;
  reg0 = 0.0;
  regw = 0.0;
  regv = 0.0;
  k0 = true;
  k1 = true;
}

void fm_model::debug() {
  std::cout << "num_attributes=" << num_attribute << std::endl;
  std::cout << "use w0=" << k0 << std::endl;
  std::cout << "use w1=" << k1 << std::endl;
  std::cout << "dim v =" << num_factor << std::endl;
  std::cout << "reg_w0=" << reg0 << std::endl;
  std::cout << "reg_w=" << regw << std::endl;
  std::cout << "reg_v=" << regv << std::endl;
  std::cout << "init ~ N(" << init_mean << "," << init_stdev << ")" << std::endl;
}

void fm_model::init() {
  w0 = 0; // 常数项的系数
  w.setSize(num_attribute); // 设置一次项系数的个数
  v.setSize(num_factor, num_attribute); // 设置交叉项的矩阵大小
  w.init(0);  // 初始化一次项系数为0
  v.init(init_mean, init_stdev);  // 按照均值和方差初始化交叉项系数
  // 交叉项中的两个参数，设置其大小为num_factor
  m_sum.setSize(num_factor);
  m_sum_sqr.setSize(num_factor);
}

double fm_model::predict(sparse_row<FM_FLOAT>& x) {
  return predict(x, m_sum, m_sum_sqr);
}

double fm_model::predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr) {
  double result = 0;  // 最终的结果
  // 第一部分 常数项
  if (k0) {
    result += w0;
  }
  // 第二部分 一次项
  if (k1) {
    for (uint i = 0; i < x.size; i++) {
      assert(x.data[i].id < num_attribute);
      result += w(x.data[i].id) * x.data[i].value;
    }
  }
  // 第三部分
  // 交叉项，对应着化简后的公式，有两重循环
  for (int f = 0; f < num_factor; f++) {
    sum(f) = 0;
    sum_sqr(f) = 0;
    for (uint i = 0; i < x.size; i++) {
      double d = v(f,x.data[i].id) * x.data[i].value;
      sum(f) += d;
      sum_sqr(f) += d*d;
    }
    result += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
  }
  return result;
}

/*
 * Write the FM model (all the parameters) in a file.
 */
void fm_model::saveModel(std::string model_file_path){
  std::ofstream out_model;
  out_model.open(model_file_path.c_str());
  if (k0) {
    out_model << "#global bias W0" << std::endl;
    out_model << w0 << std::endl;
  }
  if (k1) {
    out_model << "#unary interactions Wj" << std::endl;
    for (uint i = 0; i<num_attribute; i++){
      out_model <<  w(i) << std::endl;
    }
  }
  out_model << "#pairwise interactions Vj,f" << std::endl;
  for (uint i = 0; i<num_attribute; i++){
    for (int f = 0; f < num_factor; f++) {
      out_model << v(f,i);
      if (f!=num_factor-1){ out_model << ' '; }
    }
    out_model << std::endl;
  }
  out_model.close();
}

/*
 * Read the FM model (all the parameters) from a file.
 * If no valid conversion could be performed, the function std::atof returns zero (0.0).
 */
int fm_model::loadModel(std::string model_file_path) {
  std::string line;
  std::ifstream model_file (model_file_path.c_str());
  if (model_file.is_open()){
    if (k0) {
      if(!std::getline(model_file,line)){return 0;} // "#global bias W0"
      if(!std::getline(model_file,line)){return 0;}
      w0 = std::atof(line.c_str());
    }
    if (k1) {
      if(!std::getline(model_file,line)){return 0;} //"#unary interactions Wj"
      for (uint i = 0; i<num_attribute; i++){
        if(!std::getline(model_file,line)){return 0;}
        w(i) = std::atof(line.c_str());
      }
    }
    if(!std::getline(model_file,line)){return 0;}; // "#pairwise interactions Vj,f"
    for (uint i = 0; i<num_attribute; i++){
      if(!std::getline(model_file,line)){return 0;}
      std::vector<std::string> v_str;
      splitString(line, ' ', v_str);
      if ((int)v_str.size() != num_factor){return 0;}
      for (int f = 0; f < num_factor; f++) {
        v(f,i) = std::atof(v_str[f].c_str());
      }
    }
    model_file.close();
  }
  else{ return 0;}
  return 1;
}

/*
 * Splits the string s around matches of the given character c, and stores the substrings in the vector v
 */
void fm_model::splitString(const std::string& s, char c, std::vector<std::string>& v) {
  std::string::size_type i = 0;
  std::string::size_type j = s.find(c);
  while (j != std::string::npos) {
    v.push_back(s.substr(i, j-i));
    i = ++j;
    j = s.find(c, j);
    if (j == std::string::npos)
      v.push_back(s.substr(i, s.length()));
  }
}

#endif /*FM_MODEL_H_*/
