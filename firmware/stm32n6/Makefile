# TinyBench-KWS STM32N6570-DK firmware — CPU path (X-CUBE-AI ai_network_* runtime)
#
#   make MODEL=dscnn      # or tcresnet, gru
#   make MODEL=dscnn flash-info   # show how to load via ST-Link
#   make clean
#
# Produces build/$(MODEL)_cpu/Project.elf (+ .bin, .hex).

MODEL ?= dscnn

# ── Toolchain ───────────────────────────────────────────────────────────────
GCC_BIN := $(firstword $(wildcard /Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin))
CC  := $(GCC_BIN)/arm-none-eabi-gcc
CP  := $(GCC_BIN)/arm-none-eabi-objcopy
SZ  := $(GCC_BIN)/arm-none-eabi-size

# ── Paths ───────────────────────────────────────────────────────────────────
PROJ    := $(CURDIR)
SDK     := /Users/rohini/Downloads/stedgeai/Projects/STM32N6570-DK/Applications/CM55_Validation
HALDIR  := $(SDK)/Drivers/STM32N6xx_HAL_Driver
CMSIS   := $(SDK)/Drivers/CMSIS
SYSDIR  := $(CMSIS)/Device/ST/STM32N6xx/Source/Templates
AI      := /Users/rohini/Downloads/stedgeai/Middlewares/ST/AI
MODELDIR := $(PROJ)/kws_$(MODEL)_cpu
BUILD   := $(PROJ)/build/$(MODEL)_cpu

# ── Per-model defines ────────────────────────────────────────────────────────
# dscnn_v2 (HW-friendly 25x6) input quant: scale 0.04317872, zp 11 (was 0.043562427/3 for original 50x11).
DEFS_dscnn    := -DAI_NETWORK_IN_SCALE=0.04317872226238251f -DAI_NETWORK_IN_ZP=11
DEFS_tcresnet :=
DEFS_gru      :=
MODEL_DEFS := $(DEFS_$(MODEL))

# ── Sources ─────────────────────────────────────────────────────────────────
APP_SRC := \
  $(PROJ)/src/main.c \
  $(PROJ)/src/kws_bench.c \
  $(PROJ)/src/stm32n6xx_it.c \
  $(PROJ)/src/system_stm32n6xx.c \
  $(PROJ)/src/syscalls.c

MODEL_SRC := \
  $(MODELDIR)/network.c \
  $(MODELDIR)/network_data.c \
  $(MODELDIR)/network_data_params.c

HAL_SRC := \
  $(HALDIR)/Src/stm32n6xx_hal.c \
  $(HALDIR)/Src/stm32n6xx_hal_cortex.c \
  $(HALDIR)/Src/stm32n6xx_hal_cacheaxi.c \
  $(HALDIR)/Src/stm32n6xx_hal_gpio.c \
  $(HALDIR)/Src/stm32n6xx_hal_pwr.c \
  $(HALDIR)/Src/stm32n6xx_hal_pwr_ex.c \
  $(HALDIR)/Src/stm32n6xx_hal_rcc.c \
  $(HALDIR)/Src/stm32n6xx_hal_rcc_ex.c \
  $(HALDIR)/Src/stm32n6xx_hal_uart.c \
  $(HALDIR)/Src/stm32n6xx_hal_uart_ex.c

ASM_SRC := $(PROJ)/bsp/startup_stm32n657xx.s

C_SRC := $(APP_SRC) $(MODEL_SRC) $(HAL_SRC)

# ── Includes ────────────────────────────────────────────────────────────────
INC := \
  -I$(PROJ)/src \
  -I$(PROJ)/bsp \
  -I$(MODELDIR) \
  -I$(AI)/Inc \
  -I$(HALDIR)/Inc \
  -I$(CMSIS)/Core/Include \
  -I$(CMSIS)/Device/ST/STM32N6xx/Include

# ── Flags ───────────────────────────────────────────────────────────────────
CPU   := -mcpu=cortex-m55 -mcmse -mthumb
FPU   := -mfpu=auto -mfloat-abi=hard
DEFS  := -DSTM32N657xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSER_VECT_TAB_ADDRESS -DVECT_TAB_SRAM \
         -DBENCH_NPU=0 -DBENCH_MODEL_STR='"$(MODEL)"' $(MODEL_DEFS)
OPT   := -O2 -g3
CFLAGS := $(CPU) $(FPU) $(DEFS) $(INC) $(OPT) -std=c11 -Wall \
          -fdata-sections -ffunction-sections -MMD -MP

LDSCRIPT := $(PROJ)/bsp/STM32N657xx.ld
LIBDIR   := $(AI)/Lib/GCC/ARMCortexM55
LIBS     := -lc -lm -lnosys -l:NetworkRuntime1020_CM55_GCC.a
LDFLAGS  := $(CPU) $(FPU) -specs=nano.specs -T$(LDSCRIPT) -L$(LIBDIR) \
            -Wl,--gc-sections -Wl,-Map=$(BUILD)/Project.map,--cref \
            -u _printf_float $(LIBS) -Wl,--print-memory-usage

# ── Objects ─────────────────────────────────────────────────────────────────
OBJ := $(addprefix $(BUILD)/obj/, $(notdir $(C_SRC:.c=.o))) \
       $(addprefix $(BUILD)/obj/, $(notdir $(ASM_SRC:.s=.o)))
VPATH := $(sort $(dir $(C_SRC) $(ASM_SRC)))

# ── Rules ───────────────────────────────────────────────────────────────────
all: $(BUILD)/Project.elf $(BUILD)/Project.bin $(BUILD)/Project.hex

$(BUILD)/obj/%.o: %.c | $(BUILD)/obj
	$(CC) -c $(CFLAGS) $< -o $@

$(BUILD)/obj/%.o: %.s | $(BUILD)/obj
	$(CC) -c $(CPU) $(FPU) -x assembler-with-cpp $< -o $@

$(BUILD)/Project.elf: $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD)/Project.bin: $(BUILD)/Project.elf
	$(CP) -O binary -S $< $@

$(BUILD)/Project.hex: $(BUILD)/Project.elf
	$(CP) -O ihex $< $@

$(BUILD)/obj:
	mkdir -p $@

clean:
	rm -rf $(PROJ)/build/$(MODEL)_cpu

flash-info:
	@echo "Load to SRAM @0x34000000 via ST-Link (board has no internal flash):"
	@echo "  STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -d $(BUILD)/Project.bin 0x34000000 -s 0x34000000"

-include $(OBJ:.o=.d)
.PHONY: all clean flash-info
