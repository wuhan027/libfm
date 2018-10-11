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
// fm_learn.h: Generic learning method for factorization machines
// 机器学习算法实现解析——libFM之libFM的训练过程概述 https://blog.csdn.net/google19890102/article/details/72866320

#ifndef FM_LEARN_H_
#define FM_LEARN_H_

#include <cmath>
#include "Data.h"
#include "../../fm_core/fm_model.h"
#include "../../util/rlog.h"
#include "../../util/util.h"

class fm_learn {
 public:
  fm_learn();
  virtual void init();
  virtual double evaluate(Data& data);
  virtual void learn(Data& train, Data& test);
  virtual void predict(Data& data, DVector<double>& out) = 0;
  virtual void debug();

  DataMetaInfo* meta;
  fm_model* fm; // 对应的fm模型
  double min_target;  // 设置的预测值的最小值
  double max_target;  // 设置的预测值的最大值

    // task用于区分不同的任务：0表示的是回归，1表示的是分类
  int task; // 0=regression, 1=classification
    // 定义两个常量，分别表示的是回归和分类
  const static int TASK_REGRESSION = 0;
  const static int TASK_CLASSIFICATION = 1;

  Data* validation; // 验证数据集
  RLog* log;  // 日志指针

 protected:
  // these functions can be overwritten (e.g. for MCMC)
  virtual double evaluate_classification(Data& data);
  virtual double evaluate_regression(Data& data);
  virtual double predict_case(Data& data);

  DVector<double> sum, sum_sqr; // FM模型的交叉项中的两项
  DMatrix<double> pred_q_term;
};

// Implementation
double fm_learn::predict_case(Data& data) {
  return fm->predict(data.data->getRow());
}

// 构造函数，初始化变量，实例化的过程在main函数中
fm_learn::fm_learn() {
  log = NULL;
  task = 0;
  meta = NULL;
}

void fm_learn::init() {
  // 日志
  if (log != NULL) {
    if (task == TASK_REGRESSION) {
      log->addField("rmse", std::numeric_limits<double>::quiet_NaN());
      log->addField("mae", std::numeric_limits<double>::quiet_NaN());
    } else if (task == TASK_CLASSIFICATION) {
      log->addField("accuracy", std::numeric_limits<double>::quiet_NaN());
    } else {
      throw "unknown task";
    }
    log->addField("time_pred", std::numeric_limits<double>::quiet_NaN());
    log->addField("time_learn", std::numeric_limits<double>::quiet_NaN());
    log->addField("time_learn2", std::numeric_limits<double>::quiet_NaN());
    log->addField("time_learn4", std::numeric_limits<double>::quiet_NaN());
  }
  // 设置交叉项中的两项的大小
  sum.setSize(fm->num_factor);
  sum_sqr.setSize(fm->num_factor);
  pred_q_term.setSize(fm->num_factor, meta->num_relations + 1); //?
}

// 对数据的评估
double fm_learn::evaluate(Data& data) {
  assert(data.data != NULL);
  if (task == TASK_REGRESSION) {
    return evaluate_regression(data);
  } else if (task == TASK_CLASSIFICATION) {
    return evaluate_classification(data);
  } else {
    throw "unknown task";
  }
}

// 模型的训练过程
void fm_learn::learn(Data& train, Data& test) {
}

// debug函数，用于打印中间的结果
void fm_learn::debug() {
  std::cout << "task=" << task << std::endl;
  std::cout << "min_target=" << min_target << std::endl;
  std::cout << "max_target=" << max_target << std::endl;
}

// 对分类问题的评价
double fm_learn::evaluate_classification(Data& data) {
  int num_correct = 0;  // 准确类别的个数
  double eval_time = getusertime();
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);  // 对样本进行预测
    // 利用预测值的符号与原始标签值的符号是否相同，若相同，则预测是准确的
    if (((p >= 0) && (data.target(data.data->getRowIndex()) >= 0)) || ((p < 0) && (data.target(data.data->getRowIndex()) < 0))) {
      num_correct++;
    }
  }
  eval_time = (getusertime() - eval_time);
  // log the values
  if (log != NULL) {
    log->log("accuracy", (double) num_correct / (double) data.data->getNumRows());
    log->log("time_pred", eval_time);
  }

  return (double) num_correct / (double) data.data->getNumRows();
}

// 对回归问题的评价
double fm_learn::evaluate_regression(Data& data) {
  double rmse_sum_sqr = 0;  // 误差的平方和
  double mae_sum_abs = 0; // 误差的绝对值之和
  double eval_time = getusertime();
  // 取出每一条样本
  int i = 0;
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);  // 计算该样本的预测值
    if (i == 0) {
      std::cout<<"i===" <<i <<",p===" << p <<std::endl;
    }
    i++;
    p = std::min(max_target, p);  // 防止预测值超出最大限制
    p = std::max(min_target, p);  // 防止预测值超出最小限制
    double err = p - data.target(data.data->getRowIndex()); // 得到预测值与真实值之间的误差
    rmse_sum_sqr += err*err;  // 计算误差平方和
    mae_sum_abs += std::abs((double)err); // 计算误差绝对值之和
  }
  eval_time = (getusertime() - eval_time);
  // log the values
  if (log != NULL) {
    log->log("rmse", std::sqrt(rmse_sum_sqr/data.data->getNumRows()));
    log->log("mae", mae_sum_abs/data.data->getNumRows());
    log->log("time_pred", eval_time);
  }

  return std::sqrt(rmse_sum_sqr/data.data->getNumRows()); // 返回均方根误差
}

#endif /*FM_LEARN_H_*/
