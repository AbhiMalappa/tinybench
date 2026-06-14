/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-13T09:41:13-0600
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
#define AI_NETWORK_MODEL_SIGNATURE     "0xbe3178c3bd61ceae34ad51c6f9f354ac"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-06-13T09:41:13-0600"

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
  mfcc_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 490, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _Squeeze_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 490, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 570, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 784, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 912, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 600, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 792, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_relu_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 792, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 416, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 672, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 416, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 416, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_relu_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 416, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 672, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 336, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 720, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 336, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 336, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_relu_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 336, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 35, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1440, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3456, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5184, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13824, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20736, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1680, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 35, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3464, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6032, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6320, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1168, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6432, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6720, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2080, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6944, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7520, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3872, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 223, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Squeeze_output_0_to_chlast_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043562427163124084f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021268188953399658f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020019670948386192f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0012067427160218358f, 0.0031169699504971504f, 0.0015073346439749002f, 0.0015325378626585007f, 0.002524246694520116f, 0.002362603321671486f, 0.00184069131501019f, 0.0012211314169690013f, 0.0019092370057478547f, 0.0020112288184463978f, 0.0014096839586272836f, 0.001452524564228952f, 0.0024983701296150684f, 0.0010375978890806437f, 0.0015672520967200398f, 0.002350626280531287f, 0.0017764049116522074f, 0.00225852127186954f, 0.0018900310387834907f, 0.0025690740440040827f, 0.003803981700912118f, 0.0016179407248273492f, 0.0022427053190767765f, 0.004174729343503714f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_3_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05915888771414757f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021268188953399658f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_main_main_3_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004967450629919767f, 0.0027547364588826895f, 0.003824351355433464f, 0.008127751760184765f, 0.0035463336389511824f, 0.004672320559620857f, 0.0022419767919927835f, 0.00296921469271183f, 0.002774956403300166f, 0.0032933775801211596f, 0.004528629127889872f, 0.0066743590869009495f, 0.003004607744514942f, 0.0024059268180280924f, 0.004050113260746002f, 0.002961535705253482f, 0.003180449130013585f, 0.003473724937066436f, 0.00434398278594017f, 0.006527199875563383f, 0.003781981533393264f, 0.003411607351154089f, 0.0037452655378729105f, 0.004878601990640163f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_relu_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028944898396730423f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_skip_skip_0_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04538780823349953f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002339918166399002f, 0.007714162115007639f, 0.006871482823044062f, 0.0014688248047605157f, 0.00492382375523448f, 0.008355401456356049f, 0.002736105117946863f, 0.003579722950235009f, 0.0033381925895810127f, 0.00416663708165288f, 0.0031987179536372423f, 0.004572413861751556f, 0.0042848712764680386f, 0.004693598952144384f, 0.007010309956967831f, 0.004079167731106281f, 0.004356022458523512f, 0.004653571639209986f, 0.00353618455119431f, 0.0029581296257674694f, 0.0021681084763258696f, 0.0039222328923642635f, 0.006255860906094313f, 0.004887842107564211f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01784517988562584f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028944898396730423f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0012958942679688334f, 0.0009302884573116899f, 0.0016950896242633462f, 0.0016602084506303072f, 0.001565866288729012f, 0.0015460135182365775f, 0.0017106959130614996f, 0.0012010872596874833f, 0.0012283780379220843f, 0.0019723870791494846f, 0.00173225870821625f, 0.0016196654178202152f, 0.0019513654988259077f, 0.001558836898766458f, 0.0019181926036253572f, 0.0014709167880937457f, 0.0015652498695999384f, 0.0012804241850972176f, 0.0014729960821568966f, 0.0017589963972568512f, 0.0015731335151940584f, 0.0019640650134533644f, 0.0013552185846492648f, 0.002676808973774314f, 0.001606848556548357f, 0.0017252953257411718f, 0.0017423860263079405f, 0.001906771445646882f, 0.0016288557089865208f, 0.0012518640141934156f, 0.001452837255783379f, 0.00174501643050462f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_3_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03662247955799103f),
    AI_PACK_INTQ_ZP(20)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01784517988562584f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_main_main_3_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002701790537685156f, 0.0024288224522024393f, 0.0024482514709234238f, 0.002895657904446125f, 0.0028042157646268606f, 0.002971218666061759f, 0.0027481215074658394f, 0.0025378779973834753f, 0.0032669806387275457f, 0.0032185488380491734f, 0.002728302963078022f, 0.0025222564581781626f, 0.0020123522263020277f, 0.0019091469002887607f, 0.002812902443110943f, 0.0031889951787889004f, 0.0028056236915290356f, 0.001932367100380361f, 0.00388354598544538f, 0.0026692443061619997f, 0.0023466970305889845f, 0.0036539509892463684f, 0.0028721222188323736f, 0.002849364187568426f, 0.0015710227889940143f, 0.0023684282787144184f, 0.003918015863746405f, 0.0018454536329954863f, 0.002770858583971858f, 0.003197688842192292f, 0.0032902711536735296f, 0.0023681456223130226f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_relu_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02246873266994953f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_skip_skip_0_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021036392077803612f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00185330247040838f, 0.0015810485929250717f, 0.0017157067777588964f, 0.0012411157367751002f, 0.0018221404170617461f, 0.001247782027348876f, 0.0016744137974455953f, 0.002076000440865755f, 0.0018773761112242937f, 0.003032238222658634f, 0.0022206916473805904f, 0.002293539000675082f, 0.0018827383173629642f, 0.0008747523534111679f, 0.0016617145156487823f, 0.002119333017617464f, 0.002025705762207508f, 0.0016987683484330773f, 0.0030960857402533293f, 0.0013632680056616664f, 0.0012361831031739712f, 0.0010667649330571294f, 0.001071440288797021f, 0.0010337188141420484f, 0.0011331336572766304f, 0.0014045328134670854f, 0.001496864133514464f, 0.0007662909920327365f, 0.0016887009842321277f, 0.0015783576527610421f, 0.001440694322809577f, 0.0009014092502184212f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01040517259389162f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02246873266994953f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018148499075323343f, 0.0021131366956979036f, 0.0018743708496913314f, 0.001483358209952712f, 0.001402778667397797f, 0.002041708445176482f, 0.0019136094488203526f, 0.0013693489599972963f, 0.001930922968313098f, 0.0018370590405538678f, 0.002024752786383033f, 0.002531761769205332f, 0.004629299510270357f, 0.0019928941037505865f, 0.002197509165853262f, 0.0016695103840902448f, 0.0014687977964058518f, 0.0012929672375321388f, 0.002089496236294508f, 0.0015576580772176385f, 0.0016673371428623796f, 0.0027796432841569185f, 0.0017233426915481687f, 0.001996566541492939f, 0.0014410268049687147f, 0.0016404703492298722f, 0.001717757317237556f, 0.0014930516481399536f, 0.0015754305059090257f, 0.0017868091817945242f, 0.0017986585153266788f, 0.0018828024622052908f, 0.0015888480702415109f, 0.0016619412926957011f, 0.0013361433520913124f, 0.0016470793634653091f, 0.0014522927813231945f, 0.0017662437167018652f, 0.001859339070506394f, 0.001638770685531199f, 0.0013998564099892974f, 0.001424363232217729f, 0.0023784113582223654f, 0.0021294811740517616f, 0.0017936667427420616f, 0.001658864552155137f, 0.0016064489027485251f, 0.002064318163320422f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_3_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05481521040201187f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01040517259389162f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_main_main_3_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010316446423530579f, 0.01375029981136322f, 0.011470667086541653f, 0.008420681580901146f, 0.009051699191331863f, 0.014817450195550919f, 0.010448063723742962f, 0.011979423463344574f, 0.010589862242341042f, 0.011428868398070335f, 0.012023396790027618f, 0.012252187356352806f, 0.009823589585721493f, 0.009635868482291698f, 0.013176621869206429f, 0.00873362272977829f, 0.012116282247006893f, 0.00938971247524023f, 0.012104320339858532f, 0.011211831122636795f, 0.012871877290308475f, 0.010084292851388454f, 0.010159850120544434f, 0.01002319622784853f, 0.012098981998860836f, 0.012266116216778755f, 0.01116950437426567f, 0.011876796372234821f, 0.011744672432541847f, 0.01105214562267065f, 0.013222849927842617f, 0.010829064063727856f, 0.008305622264742851f, 0.010150312446057796f, 0.01134549267590046f, 0.008796563372015953f, 0.009459122084081173f, 0.009652400389313698f, 0.011487534269690514f, 0.012580452486872673f, 0.008859246037900448f, 0.010610366240143776f, 0.012878172099590302f, 0.011306153610348701f, 0.012009749189019203f, 0.009394918568432331f, 0.010116412304341793f, 0.011349818669259548f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_relu_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031619057059288025f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_skip_skip_0_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03207770362496376f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007625224534422159f, 0.006371454801410437f, 0.006254879757761955f, 0.00425986060872674f, 0.004616391845047474f, 0.00568575132638216f, 0.0042396667413413525f, 0.003457132261246443f, 0.0043344879522919655f, 0.006152625195682049f, 0.005590902175754309f, 0.004264853894710541f, 0.005366167984902859f, 0.006901548709720373f, 0.005345033016055822f, 0.005554958246648312f, 0.006155683193355799f, 0.004118962213397026f, 0.008929532952606678f, 0.00695692328736186f, 0.005950241349637508f, 0.00688973581418395f, 0.006696387194097042f, 0.005850913003087044f, 0.008133326657116413f, 0.005412144120782614f, 0.006320095621049404f, 0.006604874972254038f, 0.004763063974678516f, 0.007677221670746803f, 0.005827589891850948f, 0.0043788449838757515f, 0.004104956053197384f, 0.00514794047921896f, 0.005308794789016247f, 0.004925635643303394f, 0.00614200159907341f, 0.007060051895678043f, 0.005650226026773453f, 0.005643515381962061f, 0.006462897639721632f, 0.007193315774202347f, 0.007575236260890961f, 0.007943162694573402f, 0.005989547353237867f, 0.006637473125010729f, 0.007191615644842386f, 0.0073278374038636684f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010342909954488277f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stem_stem_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020019670948386192f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stem_stem_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043562427163124084f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_stem_stem_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004461498931050301f, 0.002939240774139762f, 0.002523458795621991f, 0.0017066217260435224f, 0.0021512957755476236f, 0.0020075137726962566f, 0.0017553680809214711f, 0.005715289153158665f, 0.005152797792106867f, 0.0014110974734649062f, 0.0018439119448885322f, 0.007075151894241571f, 0.005057472735643387f, 0.0012600079644471407f, 0.0033132038079202175f, 0.0018262524390593171f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04022473469376564f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 35,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005181523039937019f, 0.00457877479493618f, 0.0051427907310426235f, 0.004960987716913223f, 0.005091828294098377f, 0.005263227038085461f, 0.004777598660439253f, 0.004542012233287096f, 0.004546644631773233f, 0.005378027446568012f, 0.004634570796042681f, 0.005001918412744999f, 0.005488904193043709f, 0.005658203735947609f, 0.004737515468150377f, 0.005291394889354706f, 0.0052854148671031f, 0.005552052985876799f, 0.005147880408912897f, 0.004973528441041708f, 0.0064625125378370285f, 0.004796994384378195f, 0.004617796279489994f, 0.005109770689159632f, 0.0046124388463795185f, 0.00532954977825284f, 0.005339271388947964f, 0.0047721839509904385f, 0.004595060832798481f, 0.005140542518347502f, 0.005355870351195335f, 0.00579344667494297f, 0.005946814548224211f, 0.005898252595216036f, 0.005706273950636387f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mfcc_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043562427163124084f),
    AI_PACK_INTQ_ZP(3)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Squeeze_output_0_to_chlast_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 49, 1), AI_STRIDE_INIT(4, 1, 1, 10, 490),
  1, &_Squeeze_output_0_to_chlast_output_array, &_Squeeze_output_0_to_chlast_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Squeeze_output_0_to_chlast_output0, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 49), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &_Squeeze_output_0_to_chlast_output_array, &_Squeeze_output_0_to_chlast_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_blocks_blocks_0_main_main_2_Relu_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 25), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_0_main_main_2_Relu_output_0_output_array, &_blocks_blocks_0_main_main_2_Relu_output_0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 57), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array, &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 6032, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6032, 6032),
  1, &_blocks_blocks_0_main_main_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_weights, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 16, 1, 9, 24), AI_STRIDE_INIT(4, 1, 16, 384, 384),
  1, &_blocks_blocks_0_main_main_2_Relu_output_0_weights_array, &_blocks_blocks_0_main_main_2_Relu_output_0_weights_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_blocks_blocks_0_main_main_3_Conv_output_0_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 25), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_0_main_main_3_Conv_output_0_output_array, &_blocks_blocks_0_main_main_3_Conv_output_0_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 33), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array, &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 6320, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6320, 6320),
  1, &_blocks_blocks_0_main_main_3_Conv_output_0_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 9, 24), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &_blocks_blocks_0_main_main_3_Conv_output_0_weights_array, &_blocks_blocks_0_main_main_3_Conv_output_0_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_relu_Relu_output_0_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 25), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_0_relu_Relu_output_0_output_array, &_blocks_blocks_0_relu_Relu_output_0_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 25), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_output_array, &_blocks_blocks_0_skip_skip_0_Conv_output_0_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 1168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1168, 1168),
  1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 16, 1, 1, 24), AI_STRIDE_INIT(4, 1, 16, 384, 384),
  1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array, &_blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_blocks_blocks_1_main_main_2_Relu_output_0_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 13), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_1_main_main_2_Relu_output_0_output_array, &_blocks_blocks_1_main_main_2_Relu_output_0_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 33), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array, &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 6432, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6432, 6432),
  1, &_blocks_blocks_1_main_main_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_weights, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 9, 32), AI_STRIDE_INIT(4, 1, 24, 768, 768),
  1, &_blocks_blocks_1_main_main_2_Relu_output_0_weights_array, &_blocks_blocks_1_main_main_2_Relu_output_0_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_blocks_blocks_1_main_main_3_Conv_output_0_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 13), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_1_main_main_3_Conv_output_0_output_array, &_blocks_blocks_1_main_main_3_Conv_output_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 21), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array, &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 6720, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6720, 6720),
  1, &_blocks_blocks_1_main_main_3_Conv_output_0_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_weights, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 9, 32), AI_STRIDE_INIT(4, 1, 32, 1024, 1024),
  1, &_blocks_blocks_1_main_main_3_Conv_output_0_weights_array, &_blocks_blocks_1_main_main_3_Conv_output_0_weights_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_relu_Relu_output_0_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 13), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_1_relu_Relu_output_0_output_array, &_blocks_blocks_1_relu_Relu_output_0_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 13), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_output_array, &_blocks_blocks_1_skip_skip_0_Conv_output_0_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 2080, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2080, 2080),
  1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_weights, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 24, 1, 1, 32), AI_STRIDE_INIT(4, 1, 24, 768, 768),
  1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array, &_blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_blocks_blocks_2_main_main_2_Relu_output_0_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 7), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_blocks_blocks_2_main_main_2_Relu_output_0_output_array, &_blocks_blocks_2_main_main_2_Relu_output_0_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 21), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array, &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 6944, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6944, 6944),
  1, &_blocks_blocks_2_main_main_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_weights, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 9, 48), AI_STRIDE_INIT(4, 1, 32, 1536, 1536),
  1, &_blocks_blocks_2_main_main_2_Relu_output_0_weights_array, &_blocks_blocks_2_main_main_2_Relu_output_0_weights_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_blocks_blocks_2_main_main_3_Conv_output_0_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 7), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_blocks_blocks_2_main_main_3_Conv_output_0_output_array, &_blocks_blocks_2_main_main_3_Conv_output_0_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 15), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array, &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_scratch0, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 7520, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7520, 7520),
  1, &_blocks_blocks_2_main_main_3_Conv_output_0_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_weights, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 48, 1, 9, 48), AI_STRIDE_INIT(4, 1, 48, 2304, 2304),
  1, &_blocks_blocks_2_main_main_3_Conv_output_0_weights_array, &_blocks_blocks_2_main_main_3_Conv_output_0_weights_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_relu_Relu_output_0_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 7), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_blocks_blocks_2_relu_Relu_output_0_output_array, &_blocks_blocks_2_relu_Relu_output_0_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_bias, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_bias_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 7), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_output_array, &_blocks_blocks_2_skip_skip_0_Conv_output_0_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 3872, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3872, 3872),
  1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_weights, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 1, 48), AI_STRIDE_INIT(4, 1, 32, 1536, 1536),
  1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array, &_blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_pool_GlobalAveragePool_output_0_output_array, &_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_bias, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_stem_stem_2_Relu_output_0_bias_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 49), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &_stem_stem_2_Relu_output_0_output_array, &_stem_stem_2_Relu_output_0_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_pad_before_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 57), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &_stem_stem_2_Relu_output_0_pad_before_output_array, &_stem_stem_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_scratch0, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 3464, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3464, 3464),
  1, &_stem_stem_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_weights, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 10, 1, 9, 16), AI_STRIDE_INIT(4, 1, 10, 160, 160),
  1, &_stem_stem_2_Relu_output_0_weights_array, &_stem_stem_2_Relu_output_0_weights_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 35, 1, 1), AI_STRIDE_INIT(4, 4, 4, 140, 140),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 35, 1, 1), AI_STRIDE_INIT(4, 1, 1, 35, 35),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 223, 1, 1), AI_STRIDE_INIT(4, 2, 2, 446, 446),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 48, 35, 1, 1), AI_STRIDE_INIT(4, 1, 48, 1680, 1680),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  mfcc_0_conversion_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 10, 49), AI_STRIDE_INIT(4, 1, 1, 1, 10),
  1, &mfcc_0_conversion_output_array, &mfcc_0_conversion_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  mfcc_output, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 10, 49), AI_STRIDE_INIT(4, 4, 4, 4, 40),
  1, &mfcc_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 72,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_layer, 66,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_pool_GlobalAveragePool_output_0_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 7), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 7), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_relu_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_blocks_blocks_2_main_main_3_Conv_output_0_output, &_blocks_blocks_2_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_relu_Relu_output_0_layer, 63,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_blocks_blocks_2_relu_Relu_output_0_chain,
  NULL, &_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_2_skip_skip_0_Conv_output_0_weights, &_blocks_blocks_2_skip_skip_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_skip_skip_0_Conv_output_0_layer, 55,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_2_skip_skip_0_Conv_output_0_chain,
  NULL, &_blocks_blocks_2_relu_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_2_main_main_3_Conv_output_0_weights, &_blocks_blocks_2_main_main_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_3_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_layer, 60,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_2_main_main_3_Conv_output_0_chain,
  NULL, &_blocks_blocks_2_skip_skip_0_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_value_data, _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_layer, 60,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_chain,
  NULL, &_blocks_blocks_2_main_main_3_Conv_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_2_main_main_2_Relu_output_0_weights, &_blocks_blocks_2_main_main_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_layer, 54,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_2_main_main_2_Relu_output_0_chain,
  NULL, &_blocks_blocks_2_main_main_3_Conv_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_value_data, _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_layer, 54,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_chain,
  NULL, &_blocks_blocks_2_main_main_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_relu_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_blocks_blocks_1_main_main_3_Conv_output_0_output, &_blocks_blocks_1_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_relu_Relu_output_0_layer, 51,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_blocks_blocks_1_relu_Relu_output_0_chain,
  NULL, &_blocks_blocks_2_main_main_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_1_skip_skip_0_Conv_output_0_weights, &_blocks_blocks_1_skip_skip_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_skip_skip_0_Conv_output_0_layer, 43,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_1_skip_skip_0_Conv_output_0_chain,
  NULL, &_blocks_blocks_1_relu_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_1_main_main_3_Conv_output_0_weights, &_blocks_blocks_1_main_main_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_3_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_layer, 48,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_1_main_main_3_Conv_output_0_chain,
  NULL, &_blocks_blocks_1_skip_skip_0_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_value_data, _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_layer, 48,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_chain,
  NULL, &_blocks_blocks_1_main_main_3_Conv_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_1_main_main_2_Relu_output_0_weights, &_blocks_blocks_1_main_main_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_layer, 42,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_1_main_main_2_Relu_output_0_chain,
  NULL, &_blocks_blocks_1_main_main_3_Conv_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_value_data, _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_layer, 42,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_chain,
  NULL, &_blocks_blocks_1_main_main_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_relu_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_blocks_blocks_0_main_main_3_Conv_output_0_output, &_blocks_blocks_0_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_relu_Relu_output_0_layer, 39,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_blocks_blocks_0_relu_Relu_output_0_chain,
  NULL, &_blocks_blocks_1_main_main_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_0_skip_skip_0_Conv_output_0_weights, &_blocks_blocks_0_skip_skip_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_skip_skip_0_Conv_output_0_layer, 31,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_0_skip_skip_0_Conv_output_0_chain,
  NULL, &_blocks_blocks_0_relu_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_0_main_main_3_Conv_output_0_weights, &_blocks_blocks_0_main_main_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_3_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_layer, 36,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_0_main_main_3_Conv_output_0_chain,
  NULL, &_blocks_blocks_0_skip_skip_0_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_value_data, _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_layer, 36,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_chain,
  NULL, &_blocks_blocks_0_main_main_3_Conv_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_blocks_blocks_0_main_main_2_Relu_output_0_weights, &_blocks_blocks_0_main_main_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_layer, 30,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_blocks_blocks_0_main_main_2_Relu_output_0_chain,
  NULL, &_blocks_blocks_0_main_main_3_Conv_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_value_data, _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_layer, 30,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_chain,
  NULL, &_blocks_blocks_0_main_main_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stem_stem_2_Relu_output_0_weights, &_stem_stem_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_layer, 27,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_stem_stem_2_Relu_output_0_chain,
  NULL, &_blocks_blocks_0_main_main_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _stem_stem_2_Relu_output_0_pad_before_value_data[] = { 3 };
AI_ARRAY_OBJ_DECLARE(
    _stem_stem_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _stem_stem_2_Relu_output_0_pad_before_value_data, _stem_stem_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Squeeze_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_pad_before_layer, 27,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_stem_stem_2_Relu_output_0_pad_before_chain,
  NULL, &_stem_stem_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_stem_stem_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 4, 0, 4, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Squeeze_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mfcc_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Squeeze_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Squeeze_output_0_to_chlast_layer, 3,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Squeeze_output_0_to_chlast_chain,
  NULL, &_stem_stem_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mfcc_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mfcc_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mfcc_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mfcc_0_conversion_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &mfcc_0_conversion_chain,
  NULL, &_Squeeze_output_0_to_chlast_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 66588, 1, 1),
    66588, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 8992, 1, 1),
    8992, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &mfcc_0_conversion_layer, 0x838c2ea9, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 66588, 1, 1),
      66588, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 8992, 1, 1),
      8992, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &mfcc_0_conversion_layer, 0x838c2ea9, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    mfcc_output_array.data = AI_PTR(g_network_activations_map[0] + 2460);
    mfcc_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2460);
    mfcc_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 1968);
    mfcc_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1968);
    _Squeeze_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 1476);
    _Squeeze_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1476);
    _stem_stem_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 1968);
    _stem_stem_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1968);
    _stem_stem_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2540);
    _stem_stem_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2540);
    _stem_stem_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1184);
    _stem_stem_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1184);
    _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 272);
    _blocks_blocks_0_main_main_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 272);
    _blocks_blocks_0_main_main_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_main_main_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_main_main_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 248);
    _blocks_blocks_0_main_main_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 248);
    _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    _blocks_blocks_0_main_main_3_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    _blocks_blocks_0_main_main_3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_main_main_3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_main_main_3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _blocks_blocks_0_main_main_3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1968);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3136);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3136);
    _blocks_blocks_0_relu_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 32);
    _blocks_blocks_0_relu_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32);
    _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 7496);
    _blocks_blocks_1_main_main_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7496);
    _blocks_blocks_1_main_main_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1064);
    _blocks_blocks_1_main_main_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1064);
    _blocks_blocks_1_main_main_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_main_main_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 7616);
    _blocks_blocks_1_main_main_3_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7616);
    _blocks_blocks_1_main_main_3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_main_main_3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_main_main_3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 7584);
    _blocks_blocks_1_main_main_3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7584);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 632);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _blocks_blocks_1_relu_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _blocks_blocks_1_relu_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 7616);
    _blocks_blocks_2_main_main_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7616);
    _blocks_blocks_2_main_main_2_Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_main_main_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_main_main_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 7568);
    _blocks_blocks_2_main_main_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7568);
    _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_main_main_3_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_main_main_3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1136);
    _blocks_blocks_2_main_main_3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1136);
    _blocks_blocks_2_main_main_3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8656);
    _blocks_blocks_2_main_main_3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8656);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 416);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4288);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4288);
    _blocks_blocks_2_relu_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _blocks_blocks_2_relu_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 336);
    _pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 336);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 384);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
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
    
    _stem_stem_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stem_stem_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _stem_stem_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _stem_stem_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stem_stem_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1440);
    _stem_stem_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1440);
    _blocks_blocks_0_main_main_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_main_main_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1504);
    _blocks_blocks_0_main_main_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1504);
    _blocks_blocks_0_main_main_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_main_main_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4960);
    _blocks_blocks_0_main_main_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4960);
    _blocks_blocks_0_main_main_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_main_main_3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5056);
    _blocks_blocks_0_main_main_3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5056);
    _blocks_blocks_0_main_main_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_main_main_3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10240);
    _blocks_blocks_0_main_main_3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10240);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10336);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10336);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_0_skip_skip_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10720);
    _blocks_blocks_0_skip_skip_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10720);
    _blocks_blocks_1_main_main_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_main_main_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10816);
    _blocks_blocks_1_main_main_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10816);
    _blocks_blocks_1_main_main_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_main_main_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 17728);
    _blocks_blocks_1_main_main_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 17728);
    _blocks_blocks_1_main_main_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_main_main_3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 17856);
    _blocks_blocks_1_main_main_3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 17856);
    _blocks_blocks_1_main_main_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_main_main_3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 27072);
    _blocks_blocks_1_main_main_3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 27072);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 27200);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 27200);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_1_skip_skip_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 27968);
    _blocks_blocks_1_skip_skip_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 27968);
    _blocks_blocks_2_main_main_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_main_main_2_Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 28096);
    _blocks_blocks_2_main_main_2_Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 28096);
    _blocks_blocks_2_main_main_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_main_main_2_Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 41920);
    _blocks_blocks_2_main_main_2_Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 41920);
    _blocks_blocks_2_main_main_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_main_main_3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 42112);
    _blocks_blocks_2_main_main_3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 42112);
    _blocks_blocks_2_main_main_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_main_main_3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 62848);
    _blocks_blocks_2_main_main_3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 62848);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 63040);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 63040);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _blocks_blocks_2_skip_skip_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 64576);
    _blocks_blocks_2_skip_skip_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 64576);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_network_weights_map[0] + 64768);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 64768);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_network_weights_map[0] + 66448);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 66448);
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
      
      .n_macc            = 773436,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x838c2ea9,
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
      
      .n_macc            = 773436,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x838c2ea9,
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

