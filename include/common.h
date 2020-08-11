/*
 * Copyright (c) 2020, Lorna Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LENET_COMMON_H
#define LENET_COMMON_H

#include <ctime>
#include <iostream>
#include "NvInfer.h"

#define CHECK(status)                       \
  do {                                      \
    auto ret = (status);                    \
    if (ret != 0) {                         \
      std::cout << "Cuda failure: " << ret; \
      abort();                              \
    }                                       \
  } while (0)

constexpr long double operator"" _GiB(long double val) { return val * (1 << 30); }
constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KiB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val) { return val * (1 << 30); }
constexpr long long int operator"" _MiB(long long unsigned int val) { return val * (1 << 20); }
constexpr long long int operator"" _KiB(long long unsigned int val) { return val * (1 << 10); }

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kWARNING) {}

  Logger(Severity severity) : reportableSeverity(severity) {}

  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "[INTERNAL_ERROR]: ";
        break;
      case Severity::kERROR:
        std::cerr << "[ERROR]: ";
        break;
      case Severity::kWARNING:
        std::cerr << "[WARNING]: ";
        break;
      case Severity::kINFO:
        std::cout << "[INFO]: ";
        break;
      default:
        std::cout << "[UNKNOWN]: ";
        break;
    }
    std::cout << msg << std::endl;
  }

  Severity reportableSeverity{Severity::kWARNING};
};

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void printMessage(unsigned int LEVEL) {
  std::time_t now = time(0);
  std::tm* ltm = localtime(&now);
  switch (LEVEL) {
    case 0:
      std::cout << "[INFO(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << ")]: ";
      break;
    case 1:
      std::cout << "[WARNING(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << ")]: ";
      break;
    case 2:
      std::cerr << "[ERROR(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << ")]: ";
      break;
    default:
      std::cout << "[INFO(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << ")]: ";
      break;
  }
}

#endif  // LENET_COMMON_H