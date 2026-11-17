#pragma once

#include <stdint.h>

// B080 op_common/log/log.h stopped exposing the unqualified OP module id used
// by inherited ops-transformer tiling/error headers. Include the CANN log type
// header early so OP still comes from the active CANN version.
#if defined(__has_include)
#if __has_include("base/log_types.h")
#include "base/log_types.h"
#elif __has_include("toolchain/log_types.h")
#include "toolchain/log_types.h"
#endif
#endif

#if !defined(LOG_TYPES_H_) && !defined(OP)
#define OP 63
#endif

#if defined(LOG_CPP) && !defined(DLOG_PUB_H_)
#ifdef __cplusplus
extern "C" {
#endif
int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...);
#ifdef __cplusplus
}
#endif
#define DLOG_PUB_H_
#endif

// CANN 9.0.1 log.h does not provide the EZ00xx helpers that vllm-ascend main
// tiling code calls. These only run on the invalid-argument path. Guarded so a
// newer CANN that already defines them keeps its own reporting.
#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entityName, paramName, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s is invalid. Reason: %s.", (paramName), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(entityName, paramName, incorrectShape, correctShape) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect shape %s, expected %s.", (paramName), (incorrectShape), (correctShape))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entityName, paramName, incorrectShape, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect shape %s. Reason: %s.", (paramName), (incorrectShape), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entityName, paramNames, incorrectShapes, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect shapes %s. Reason: %s.", (paramNames), (incorrectShapes), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(entityName, paramName, incorrectDim, correctDim) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect dim %s, expected %s.", (paramName), (incorrectDim), (correctDim))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(entityName, paramName, incorrectDim, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect dim %s. Reason: %s.", (paramName), (incorrectDim), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(entityName, paramNames, incorrectDims, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect dims %s. Reason: %s.", (paramNames), (incorrectDims), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE
#define OP_LOGE_FOR_INVALID_SHAPESIZE(entityName, paramName, incorrectSize, correctSize) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect size %s, expected %s.", (paramName), (incorrectSize), (correctSize))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(entityName, paramName, incorrectSize, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect size %s. Reason: %s.", (paramName), (incorrectSize), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(entityName, paramNames, incorrectSizes, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect sizes %s. Reason: %s.", (paramNames), (incorrectSizes), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_FORMAT
#define OP_LOGE_FOR_INVALID_FORMAT(entityName, paramName, incorrectFormat, correctFormat) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect format %s, expected %s.", (paramName), (incorrectFormat), (correctFormat))
#endif
#ifndef OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON
#define OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(entityName, paramName, incorrectFormat, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect format %s. Reason: %s.", (paramName), (incorrectFormat), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(entityName, paramNames, incorrectFormats, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect formats %s. Reason: %s.", (paramNames), (incorrectFormats), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_DTYPE
#define OP_LOGE_FOR_INVALID_DTYPE(entityName, paramName, incorrectDtype, correctDtype) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect dtype %s, expected %s.", (paramName), (incorrectDtype), (correctDtype))
#endif
#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(entityName, paramName, incorrectDtype, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect dtype %s. Reason: %s.", (paramName), (incorrectDtype), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entityName, paramNames, incorrectDtypes, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect dtypes %s. Reason: %s.", (paramNames), (incorrectDtypes), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(entityName, paramName, incorrectValue, correctValue) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect value %s, expected %s.", (paramName), (incorrectValue), (correctValue))
#endif
#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, incorrectValue, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect value %s. Reason: %s.", (paramName), (incorrectValue), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entityName, paramNames, incorrectValues, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameters %s have incorrect values %s. Reason: %s.", (paramNames), (incorrectValues), (reason))
#endif
#ifndef OP_LOGE_FOR_INVALID_LISTSIZE
#define OP_LOGE_FOR_INVALID_LISTSIZE(entityName, paramName, incorrectSize, correctSize) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect list size %s, expected %s.", (paramName), (incorrectSize), (correctSize))
#endif
#ifndef OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(entityName, paramName, incorrectSize, reason) \
    OP_LOGE_WITHOUT_REPORT((entityName), "Parameter %s has incorrect list size %s. Reason: %s.", (paramName), (incorrectSize), (reason))
#endif
