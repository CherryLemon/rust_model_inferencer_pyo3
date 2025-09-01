// tensorrt_wrapper.cpp
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>

using namespace nvinfer1;

// 一个简单的 Logger 实现，用于 TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印 Error 和 Warning 级别的信息，以避免过多的日志
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

static Logger gLogger;

// 定义 C-style 结构体，用于在 Rust 和 C++ 之间传递复杂数据
extern "C" {

struct TrtBindingInfo {
    int32_t dims[8]; // Dims::MAX_DIMS is 8
    int32_t num_dims;
    DataType data_type;
    const char* name;
    bool is_input;
};

// --- 生命周期管理 ---

IRuntime* create_runtime() {
    return createInferRuntime(gLogger);
}

void destroy_runtime(IRuntime* runtime) {
    if (runtime) delete runtime;
}

ICudaEngine* deserialize_engine(IRuntime* runtime, const char* engine_data, size_t size) {
    if (!runtime) return nullptr;
    return runtime->deserializeCudaEngine(engine_data, size);
}

void destroy_engine(ICudaEngine* engine) {
    if (engine) delete engine;
}

IExecutionContext* create_execution_context(ICudaEngine* engine) {
    if (!engine) return nullptr;
    return engine->createExecutionContext();
}

void destroy_context(IExecutionContext* context) {
    if (context) delete context;
}

// --- Engine 属性查询 (使用新的API) ---

int get_num_bindings(ICudaEngine* engine) {
    return engine ? engine->getNbIOTensors() : -1;
}

bool get_binding_info(ICudaEngine* engine, int binding_index, TrtBindingInfo* info) {
    if (!engine || !info || binding_index < 0 || binding_index >= engine->getNbIOTensors()) {
        return false;
    }

    const char* tensor_name = engine->getIOTensorName(binding_index);
    if (!tensor_name) return false;

    info->name = tensor_name;
    info->data_type = engine->getTensorDataType(tensor_name);
    info->is_input = (engine->getTensorIOMode(tensor_name) == TensorIOMode::kINPUT);

    Dims dims = engine->getTensorShape(tensor_name);
    info->num_dims = dims.nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
        info->dims[i] = dims.d[i];
    }

    return true;
}

// 这些函数在新版本TensorRT中已不再适用，返回默认值或false
bool has_implicit_batch_dimension(ICudaEngine* engine) {
    // 在TensorRT 10.x中，implicit batch已被弃用
    return false;
}

int get_max_batch_size(ICudaEngine* engine) {
    // 在TensorRT 10.x中，max batch size概念已被弃用
    return -1;
}

// --- Profile 和动态 Shape 相关 (适用于 Explicit Batch) ---

int get_num_optimization_profiles(ICudaEngine* engine) {
    return engine ? engine->getNbOptimizationProfiles() : -1;
}

bool get_profile_shape(ICudaEngine* engine, int profile_index, int binding_index, int64_t* min_shape, int64_t* opt_shape, int64_t* max_shape) {
    if (!engine || binding_index < 0 || binding_index >= engine->getNbIOTensors()) return false;

    const char* tensor_name = engine->getIOTensorName(binding_index);
    if (!tensor_name) return false;

    Dims min_dims, opt_dims, max_dims;
    min_dims = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMIN);
    opt_dims = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kOPT);
    max_dims = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMAX);

    if (min_dims.nbDims < 0 || opt_dims.nbDims < 0 || max_dims.nbDims < 0) return false;

    memcpy(min_shape, min_dims.d, min_dims.nbDims * sizeof(int64_t));
    memcpy(opt_shape, opt_dims.d, opt_dims.nbDims * sizeof(int64_t));
    memcpy(max_shape, max_dims.d, max_dims.nbDims * sizeof(int64_t));

    return true;
}

bool set_binding_shape(IExecutionContext* context, int binding_index, const int64_t* shape, int num_dims) {
    if (!context) return false;

    // 获取engine引用
    auto engine = &context->getEngine();
    if (binding_index < 0 || binding_index >= engine->getNbIOTensors()) return false;

    const char* tensor_name = engine->getIOTensorName(binding_index);
    if (!tensor_name) return false;

    // 只能为输入tensor设置shape
    if (engine->getTensorIOMode(tensor_name) != TensorIOMode::kINPUT) return false;

    Dims new_dims;
    new_dims.nbDims = num_dims;
    for (int i = 0; i < num_dims; ++i) {
        new_dims.d[i] = shape[i];
    }
    return context->setInputShape(tensor_name, new_dims);
}

// --- 推理执行 (使用新的API) ---

bool execute_async_v2(IExecutionContext* context, void** bindings, void* stream_handle) {
    if (!context) return false;

    // 在TensorRT 10.x中，需要使用enqueueV3和setTensorAddress
    // 这里我们需要先设置所有tensor的地址
    auto engine = &context->getEngine();
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensor_name = engine->getIOTensorName(i);
        if (tensor_name && bindings[i]) {
            context->setTensorAddress(tensor_name, bindings[i]);
        }
    }

    return context->enqueueV3(static_cast<cudaStream_t>(stream_handle));
}

// 兼容 implicit batch - 在新版本中不再支持
bool execute_async_implicit(IExecutionContext* context, int batch_size, void** bindings, void* stream_handle) {
    // implicit batch在TensorRT 10.x中已被弃用，使用V3 API
    return execute_async_v2(context, bindings, stream_handle);
}

int32_t get_num_io_tensors(ICudaEngine* engine) {
    return engine ? engine->getNbIOTensors() : -1;
}

const char* get_io_tensor_name(ICudaEngine* engine, int32_t index) {
    return engine ? engine->getIOTensorName(index) : nullptr;
}

// 获取 Tensor Shape
bool get_tensor_shape(ICudaEngine* engine, const char* name, int64_t* shape, int32_t* num_dims) {
    if (!engine) return false;
    Dims d = engine->getTensorShape(name);
    if (d.nbDims < 0) return false;
    *num_dims = d.nbDims;
    memcpy(shape, d.d, d.nbDims * sizeof(int64_t));
    return true;
}

// 获取 Tensor 数据类型
DataType get_tensor_dtype(ICudaEngine* engine, const char* name) {
    return engine ? engine->getTensorDataType(name) : DataType::kFLOAT; // 返回默认值以防万一
}

// 获取 Tensor 模式 (Input/Output)
TensorIOMode get_tensor_mode(ICudaEngine* engine, const char* name) {
    return engine ? engine->getTensorIOMode(name) : TensorIOMode::kNONE;
}

// 获取 Profile Shape (基于 Tensor Name)
bool get_tensor_profile_shape(ICudaEngine* engine, const char* tensor_name, int profile_index, int64_t* min_shape, int64_t* opt_shape, int64_t* max_shape, int32_t* num_dims) {
    if (!engine) return false;
    Dims min_d, opt_d, max_d;
    min_d = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMIN);
    opt_d = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kOPT);
    max_d = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMAX);

    if (min_d.nbDims < 0 || opt_d.nbDims < 0 || max_d.nbDims < 0) return false;
    *num_dims = max_d.nbDims; // 假设维度一致
    memcpy(min_shape, min_d.d, min_d.nbDims * sizeof(int64_t));
    memcpy(opt_shape, opt_d.d, opt_d.nbDims * sizeof(int64_t));
    memcpy(max_shape, max_d.d, max_d.nbDims * sizeof(int64_t));
    return true;
}

// --- V3 Context API ---

// 设置输入 Tensor Shape
bool set_input_shape(IExecutionContext* context, const char* tensor_name, const int64_t* shape, int num_dims) {
    if (!context) return false;
    Dims d;
    d.nbDims = num_dims;
    memcpy(d.d, shape, num_dims * sizeof(int64_t));
    return context->setInputShape(tensor_name, d);
}

// 设置 Tensor 地址
bool set_tensor_address(IExecutionContext* context, const char* tensor_name, void* data) {
    if (!context) return false;
    return context->setTensorAddress(tensor_name, data);
}

// 执行推理 v3
bool execute_async_v3(IExecutionContext* context, void* stream_handle) {
    if (!context) return false;
    return context->enqueueV3(static_cast<cudaStream_t>(stream_handle));
}

} // extern "C"
