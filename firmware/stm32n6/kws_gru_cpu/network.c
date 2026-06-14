/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-13T09:41:04-0600
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0xa34933235afa59505778ed0235246c08"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-06-13T09:41:04-0600"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  mfcc_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 490, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4704, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_1_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 35, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2880, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27648, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_initial_h_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3360, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 35, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _gru_GRU_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 271, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_gru_GRU_output_0_1_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842867635190487f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04021161422133446f),
    AI_PACK_INTQ_ZP(-46)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 35,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008663066662847996f, 0.008240007795393467f, 0.007917112670838833f, 0.0060472371987998486f, 0.008493875153362751f, 0.007005785591900349f, 0.005938709247857332f, 0.009261802770197392f, 0.005250589456409216f, 0.006842424627393484f, 0.009455512277781963f, 0.010193165391683578f, 0.005721056368201971f, 0.01032194308936596f, 0.008735700510442257f, 0.006546565797179937f, 0.004852600861340761f, 0.005391449201852083f, 0.008841076865792274f, 0.005949754733592272f, 0.006512482650578022f, 0.008535359054803848f, 0.006193335168063641f, 0.005591246765106916f, 0.010635383427143097f, 0.009323595091700554f, 0.006816648878157139f, 0.005099118687212467f, 0.005476947408169508f, 0.007920606061816216f, 0.006171424873173237f, 0.009143924340605736f, 0.005272630136460066f, 0.0066397846676409245f, 0.00898085068911314f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_1_conversion_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_gru_GRU_output_0_1_conversion_output_array, &_gru_GRU_output_0_1_conversion_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 576, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2304, 2304),
  1, &_gru_GRU_output_0_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_initial_h, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_gru_GRU_output_0_initial_h_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_kernel, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 10, 288, 1, 1), AI_STRIDE_INIT(4, 4, 40, 11520, 11520),
  1, &_gru_GRU_output_0_kernel_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_output0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 49), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_gru_GRU_output_0_output0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_output1, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_gru_GRU_output_0_output1_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_recurrent, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 96, 288, 1, 1), AI_STRIDE_INIT(4, 4, 384, 110592, 110592),
  1, &_gru_GRU_output_0_recurrent_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _gru_GRU_output_0_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 576, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2304, 2304),
  1, &_gru_GRU_output_0_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 35, 1, 1), AI_STRIDE_INIT(4, 4, 4, 140, 140),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 35, 1, 1), AI_STRIDE_INIT(4, 1, 1, 35, 35),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 271, 1, 1), AI_STRIDE_INIT(4, 2, 2, 542, 542),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 96, 35, 1, 1), AI_STRIDE_INIT(4, 1, 96, 3360, 3360),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  mfcc_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 49, 1), AI_STRIDE_INIT(4, 4, 4, 40, 1960),
  1, &mfcc_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  mfcc_output0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 49), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &mfcc_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_gru_GRU_output_0_1_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _gru_GRU_output_0_1_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_gru_GRU_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_gru_GRU_output_0_1_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _gru_GRU_output_0_1_conversion_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_gru_GRU_output_0_1_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _gru_GRU_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mfcc_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_gru_GRU_output_0_output0, &_gru_GRU_output_0_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 6, &_gru_GRU_output_0_kernel, &_gru_GRU_output_0_recurrent, NULL, NULL, &_gru_GRU_output_0_bias, &_gru_GRU_output_0_initial_h),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_gru_GRU_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _gru_GRU_output_0_layer, 10,
  GRU_TYPE, 0x0, NULL,
  gru, forward_gru,
  &_gru_GRU_output_0_chain,
  NULL, &_gru_GRU_output_0_1_conversion_layer, AI_STATIC, 
  .n_units = 96, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = true, 
  .reset_after = true, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 128300, 1, 1),
    128300, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 23464, 1, 1),
    23464, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &_gru_GRU_output_0_layer, 0x19a70c9d, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 128300, 1, 1),
      128300, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 23464, 1, 1),
      23464, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &_gru_GRU_output_0_layer, 0x19a70c9d, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    mfcc_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    mfcc_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _gru_GRU_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1960);
    _gru_GRU_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1960);
    _gru_GRU_output_0_output0_array.data = AI_PTR(g_network_activations_map[0] + 4264);
    _gru_GRU_output_0_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 4264);
    _gru_GRU_output_0_output1_array.data = AI_PTR(g_network_activations_map[0] + 23080);
    _gru_GRU_output_0_output1_array.data_start = AI_PTR(g_network_activations_map[0] + 23080);
    _gru_GRU_output_0_1_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _gru_GRU_output_0_1_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 96);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 96);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 640);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 640);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _gru_GRU_output_0_kernel_array.format |= AI_FMT_FLAG_CONST;
    _gru_GRU_output_0_kernel_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _gru_GRU_output_0_kernel_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _gru_GRU_output_0_recurrent_array.format |= AI_FMT_FLAG_CONST;
    _gru_GRU_output_0_recurrent_array.data = AI_PTR(g_network_weights_map[0] + 11520);
    _gru_GRU_output_0_recurrent_array.data_start = AI_PTR(g_network_weights_map[0] + 11520);
    _gru_GRU_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _gru_GRU_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 122112);
    _gru_GRU_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 122112);
    _gru_GRU_output_0_initial_h_array.format |= AI_FMT_FLAG_CONST;
    _gru_GRU_output_0_initial_h_array.data = AI_PTR(g_network_weights_map[0] + 124416);
    _gru_GRU_output_0_initial_h_array.data_start = AI_PTR(g_network_weights_map[0] + 124416);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_network_weights_map[0] + 124800);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 124800);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_network_weights_map[0] + 128160);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 128160);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1508867,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x19a70c9d,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1508867,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x19a70c9d,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

