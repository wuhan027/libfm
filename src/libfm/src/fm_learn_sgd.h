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
// fm_learn_sgd.h: Stochastic Gradient Descent based learning
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.
// 机器学习算法实现解析——libFM之libFM的训练过程之SGD的方法 https://blog.csdn.net/google19890102/article/details/72866334

#ifndef FM_LEARN_SGD_H_
#define FM_LEARN_SGD_H_

#include "fm_learn.h"
#include "../../fm_core/fm_sgd.h"

// 继承自fm_learn
class fm_learn_sgd: public fm_learn {
 public:
  virtual void init();
  virtual void learn(Data& train, Data& test);
  void SGD(sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum);

  void debug();
  virtual void predict(Data& data, DVector<double>& out);

  int num_iter; // 迭代次数
  double learn_rate;  // 学习率
  DVector<double> learn_rates;  // 多个学习率
};

// Implementation
void fm_learn_sgd::init() {
  fm_learn::init();
  learn_rates.setSize(3); // 设置学习率
//  sum.setSize(fm->num_factor);
//  sum_sqr.setSize(fm->num_factor);
}

// 利用梯度下降法进行更新，具体的训练的过程在其子类中
void fm_learn_sgd::learn(Data& train, Data& test) {
  fm_learn::learn(train, test); //空函数
  // 输出运行时的参数，包括：学习率，迭代次数
  std::cout << "learnrate=" << learn_rate << std::endl;
  std::cout << "learnrates=" << learn_rates(0) << "," << learn_rates(1) << "," << learn_rates(2) << std::endl;
  std::cout << "#iterations=" << num_iter << std::endl;

  if (train.relation.dim > 0) { // 判断relation
    throw "relations are not supported with SGD";
  }
  std::cout.flush();
}

// SGD重新修正fm模型的权重
void fm_learn_sgd::SGD(sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum) {
  fm_SGD(fm, learn_rate, x, multiplier, sum); // 调用fm_sgd中的fm_SGD函数
}

// debug函数，主要用于打印中间结果
void fm_learn_sgd::debug() {
  std::cout << "num_iter=" << num_iter << std::endl;
  fm_learn::debug();
}

// 对数据进行预测
void fm_learn_sgd::predict(Data& data, DVector<double>& out) {
  assert(data.data->getNumRows() == out.dim); // 判断样本个数是否相等
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);  // 得到线性项和交叉项的和，调用的是fm_learn中的方法
    if (task == TASK_REGRESSION ) { // 回归任务
      p = std::min(max_target, p);
      p = std::max(min_target, p);
    } else if (task == TASK_CLASSIFICATION) { // 分类任务
      p = 1.0/(1.0 + exp(-p));  // Sigmoid函数处理
    } else {
      throw "task not supported";
    }
    out(data.data->getRowIndex()) = p;
  }
}


#endif /*FM_LEARN_SGD_H_*/
