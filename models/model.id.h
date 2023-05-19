#ifndef NCNN_INCLUDE_GUARD_model_id_h
#define NCNN_INCLUDE_GUARD_model_id_h
namespace model_param_id {
const int LAYER_input = 0;
const int BLOB_input = 0;
const int LAYER__conv1_Conv = 1;
const int BLOB__conv1_Conv_output_0 = 1;
const int LAYER__Relu = 2;
const int BLOB__Relu_output_0 = 2;
const int LAYER__pool_MaxPool = 3;
const int BLOB__pool_MaxPool_output_0 = 3;
const int LAYER__conv2_Conv = 4;
const int BLOB__conv2_Conv_output_0 = 4;
const int LAYER__Relu_1 = 5;
const int BLOB__Relu_1_output_0 = 5;
const int LAYER__pool_1_MaxPool = 6;
const int BLOB__pool_1_MaxPool_output_0 = 6;
const int LAYER__Flatten = 7;
const int BLOB__Flatten_output_0 = 7;
const int LAYER__fc1_Gemm = 8;
const int BLOB__fc1_Gemm_output_0 = 8;
const int LAYER__Relu_2 = 9;
const int BLOB__Relu_2_output_0 = 9;
const int LAYER__fc2_Gemm = 10;
const int BLOB__fc2_Gemm_output_0 = 10;
const int LAYER__Relu_3 = 11;
const int BLOB__Relu_3_output_0 = 11;
const int LAYER__fc3_Gemm = 12;
const int BLOB_output = 12;
} // namespace model_param_id
#endif // NCNN_INCLUDE_GUARD_model_id_h
