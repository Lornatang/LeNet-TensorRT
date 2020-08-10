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

 // Source code from https://github.com/NVIDIA/TensorRT/blob/master/samples/common/logger.h

#ifndef LENET_LOGGER_H
#define LENET_LOGGER_H

#include "logging.h"

namespace sample
{
extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);
} // namespace sample

#endif // LENET_LOGGER_H