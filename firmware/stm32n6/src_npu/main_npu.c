/*
 * TinyBench-KWS STM32N6570-DK NPU firmware entry point (Neural-ART, weights in SRAM).
 * Board init is identical to the CPU firmware; this adds the NPU hardware bring-up
 * (NPU + CACHEAXI clocks, NPU cache, RIF security, RISAF memory firewalls) so the
 * Neural-ART accelerator can run the LL_ATON network from internal npuRAM.
 * Protocol: see kws/host/protocol.md
 */
#include <string.h>
#include <stdio.h>
#include "stm32n6xx_hal.h"
#include "stm32n6xx_hal_rif.h"
#include "npu_cache.h"
#ifdef WEIGHTS_IN_FLASH
#include "stm32n6570_discovery_xspi.h"
#endif

UART_HandleTypeDef huart1;

void kws_benchmark_run(void);   /* NPU variant, in kws_bench_npu.c */

/* Debug stage marker over USART1 (huart1 ready after uart_init). */
void dbg(const char *s) { HAL_UART_Transmit(&huart1, (uint8_t *)s, strlen(s), HAL_MAX_DELAY); }

/* ── Vector table / IRQ setup ────────────────────────────────────────────── */
static void board_early_init(void)
{
    __disable_irq();
    SCB->VTOR = 0x34000000;
    __DSB();
    memset((uint32_t *)NVIC->ICER, 0xFF, sizeof(NVIC->ICER));
    memset((uint32_t *)NVIC->ICPR, 0xFF, sizeof(NVIC->ICPR));
    __enable_irq();
}

/* ── Power-domain bringup (must precede UART/GPIO use on STM32N6) ─────────── */
static void power_init(void)
{
    RCC->BUSENSR = 0xFFFFFFFF;
    HAL_PWREx_EnableVddA();
    HAL_PWREx_EnableVddIO2();
    HAL_PWREx_EnableVddIO3();
    HAL_PWREx_EnableVddIO4();
    HAL_PWREx_EnableVddIO5();
    __HAL_RCC_PWR_CLK_ENABLE();
    if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK) while (1);
    if (HAL_PWREx_GetSupplyConfig() != PWR_EXTERNAL_SOURCE_SUPPLY) while (1);
}

/* ── SRAM banks + cache-controller unlock ────────────────────────────────── */
static void memory_init(void)
{
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    __HAL_RCC_CRC_CLK_ENABLE();
    __HAL_RCC_RAMCFG_CLK_ENABLE();   /* RAMCFG must be clocked before its CR writes
                                        take effect — required to wake AXISRAM3-6 */
    RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN |
                   RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;
    RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN;
    RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

static void delay_busy(volatile uint32_t iters) { while (iters--) { __NOP(); } }

static void smps_overdrive(void)
{
    __HAL_RCC_GPIOF_CLK_ENABLE();
    GPIO_InitTypeDef g = {0};
    g.Pin = GPIO_PIN_4; g.Mode = GPIO_MODE_OUTPUT_PP; g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOF, &g);
    HAL_GPIO_WritePin(GPIOF, GPIO_PIN_4, GPIO_PIN_SET);
    delay_busy(500000);
}

/* ── System clock: PLL1 → 800 MHz CPU; IC6 (NPU/sysc_ck) from PLL1 ─────────── */
static void clock_config_800mhz(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE0) != HAL_OK) while(1);

    HAL_RCC_GetClockConfig(&clk);
    if ((clk.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
        (clk.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11)) {
        clk.ClockType   = RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK;
        clk.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
        clk.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
        if (HAL_RCC_ClockConfig(&clk) != HAL_OK) while(1);
    }

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState       = RCC_HSI_ON;
    osc.HSIDiv         = RCC_HSI_DIV1;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL1.PLLState  = RCC_PLL_ON;
    osc.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
    osc.PLL1.PLLM = 8; osc.PLL1.PLLN = 100; osc.PLL1.PLLFractional = 0;
    osc.PLL1.PLLP1 = 1; osc.PLL1.PLLP2 = 1;
    osc.PLL2.PLLState = RCC_PLL_NONE;
    osc.PLL3.PLLState = RCC_PLL_NONE;
    osc.PLL4.PLLState = RCC_PLL_NONE;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) while(1);

    clk.ClockType = RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_HCLK |
                    RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 |
                    RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_PCLK5 | RCC_CLOCKTYPE_PCLK4;
    clk.CPUCLKSource  = RCC_CPUCLKSOURCE_IC1;
    clk.SYSCLKSource  = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
    clk.AHBCLKDivider = RCC_HCLK_DIV2;
    clk.APB1CLKDivider = RCC_APB1_DIV1;
    clk.APB2CLKDivider = RCC_APB2_DIV1;
    clk.APB4CLKDivider = RCC_APB4_DIV1;
    clk.APB5CLKDivider = RCC_APB5_DIV1;
    clk.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC1Selection.ClockDivider   = 1;
    clk.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC2Selection.ClockDivider   = 2;
    clk.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC6Selection.ClockDivider   = 1;    /* NPU/sysc_ck = 800 MHz (was 200 via /4; ST runs NPU ~1GHz) */
    clk.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC11Selection.ClockDivider   = 2;
    if (HAL_RCC_ClockConfig(&clk) != HAL_OK) while(1);
}

static void uart_init(void)
{
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_USART1_FORCE_RESET();
    __HAL_RCC_USART1_RELEASE_RESET();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitTypeDef g = {0};
    g.Pin = GPIO_PIN_5; g.Mode = GPIO_MODE_AF_PP; g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW; g.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOE, &g);
    g.Pin = GPIO_PIN_6; g.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOE, &g);
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart1) != HAL_OK) while(1);
    huart1.Instance->CR1 &= ~USART_CR1_FIFOEN;
}

static void mpu_config(void)
{
    /* Leave the MPU disabled so npuRAM keeps its default Normal write-back
       cacheable attributes — exactly like ST's validation app (which configures
       no MPU). The model was generated with --cache-maintenance, so the LL_ATON
       epoch schedule issues the MCU + NPU cache clean/invalidate ops needed for
       CPU(SW-epoch)↔NPU coherency. Forcing npuRAM non-cacheable defeats those
       maintenance ops and corrupts the SW-fallback epochs' inputs. */
    HAL_MPU_Disable();
}

/* ── NPU hardware: clocks + cache only ────────────────────────────────────── */
/* NOTE: we deliberately do NOT reconfigure RISAF/RIF memory firewalls here.
   ST's NPU example does, but it runs as a signed FSBL booted through the bootrom
   with a matching security context. Our image is loaded straight into SRAM by
   the debugger, so rewriting the SRAM firewalls cut off the ST-Link's own access
   path and locked the device (only a power cycle cleared it). The bootrom already
   leaves memory permissively accessible (every CPU SRAM load proves this), so we
   keep only the minimal NPU clock + cache bring-up. */
#ifdef WEIGHTS_IN_FLASH
/* Bring the external octo-flash (MX66, XSPI2 @0x71000000) up in memory-mapped
   OPI/DTR mode so the NPU can read its weights blob directly during inference. */
static void xspi_flash_init(void)
{
    BSP_XSPI_NOR_Init_t flash;
    flash.InterfaceMode = MX66UW1G45G_OPI_MODE;
    flash.TransferRate  = MX66UW1G45G_DTR_TRANSFER;
    if (BSP_XSPI_NOR_Init(0, &flash) != BSP_ERROR_NONE) while (1);
    BSP_XSPI_NOR_EnableMemoryMappedMode(0);
}
#endif

/* Open the NPU-relevant memory firewalls so the Neural-ART master ports can
   reach npuRAM activations and (EXTMEM) the external-flash weights. In dev-boot
   the bootrom leaves these restrictive (secure CID=1 only), so the NPU's reads
   are denied and it computes on zeros → constant saturated output.
   We deliberately do NOT touch RISAF2/RISAF3 (SRAM1/SRAM2 — where the ST-Link
   loads our firmware); reconfiguring those is what locked the debugger before.
   Opening only RISAF4/5/6/12 keeps the firmware-reload path intact. */
static void risaf_open(RISAF_TypeDef *risaf, uint32_t limit)
{
    RISAF_BaseRegionConfig_t c;
    c.StartAddress   = 0x0;
    c.EndAddress     = limit;
    c.Filtering      = RISAF_FILTER_ENABLE;
    c.PrivWhitelist  = RIF_CID_NONE;
    c.ReadWhitelist  = RIF_CID_MASK;
    c.WriteWhitelist = RIF_CID_MASK;
    c.Secure = RIF_ATTRIBUTE_SEC;
    HAL_RIF_RISAF_ConfigBaseRegion(risaf, 0, &c);
    c.Secure = RIF_ATTRIBUTE_NSEC;
    HAL_RIF_RISAF_ConfigBaseRegion(risaf, 1, &c);
}

static void risaf_npu_config(void)
{
    __HAL_RCC_RIFSC_CLK_ENABLE();   /* RISAF regs fault if the security IPs aren't clocked */
    __HAL_RCC_RISAF_CLK_ENABLE();
    __HAL_RCC_NPU_CLK_ENABLE();     /* RISAF4/5 are the NPU master firewalls — NPU must be clocked */
    dbg("R0\n");
    risaf_open(RISAF4_S, RISAF4_LIMIT_ADDRESS_SPACE_SIZE);    /* NPU MST0 */
    dbg("R4\n");
    risaf_open(RISAF5_S, RISAF5_LIMIT_ADDRESS_SPACE_SIZE);    /* NPU MST1 */
    dbg("R5\n");
    risaf_open(RISAF6_S, RISAF6_LIMIT_ADDRESS_SPACE_SIZE);    /* SRAM3-6 = npuRAM (activations) */
    dbg("R6\n");
#ifdef WEIGHTS_IN_FLASH
    risaf_open(RISAF12_S, RISAF12_LIMIT_ADDRESS_SPACE_SIZE);  /* OCTOSPI2 ext flash @0x70000000 */
    dbg("R12\n");
#endif
}

static void npu_config(void)
{
    __HAL_RCC_NPU_CLK_ENABLE();
    __HAL_RCC_NPU_FORCE_RESET();
    __HAL_RCC_NPU_RELEASE_RESET();
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_FORCE_RESET();
    __HAL_RCC_CACHEAXI_RELEASE_RESET();

    /* NPU AXI cache ENABLED — the model was generated with --cache-maintenance,
       so the epoch schedule issues the NPU-cache clean/invalidate ops that keep
       the HW epochs and the CPU-run SW-fallback epochs coherent. (Disabling it
       left the SW epochs reading stale npuRAM → data-dependent but wrong output.) */
    npu_cache_init();
    npu_cache_enable();
}

int main(void)
{
    board_early_init();
    SCB_EnableICache();
    SCB_EnableDCache();
    HAL_Init();
    power_init();
    memory_init();
    __HAL_RCC_CPUCLK_CONFIG(RCC_CPUCLKSOURCE_HSI);
    __HAL_RCC_SYSCLK_CONFIG(RCC_SYSCLKSOURCE_HSI);
    smps_overdrive();
    clock_config_800mhz();
    mpu_config();
    uart_init();
    dbg("A:uart\n");

#ifdef WEIGHTS_IN_FLASH
    xspi_flash_init();   /* memory-map external flash so NPU can read weights @0x71000000 */
    dbg("X:xspi\n");
#endif

    risaf_npu_config();  /* open NPU master + npuRAM (+ flash) firewalls — NOT SRAM1/2 */
    dbg("R:risaf\n");

    npu_config();
    dbg("B:npu\n");

    /* Copy NPU weights from the firmware image (embedded .rodata blob, loaded with
       the firmware into AXISRAM1) into npuRAM5 @0x342E0000, now that memory_init()
       has enabled that bank. Embedding avoids needing a separate ST-Link staging
       load to a bank that isn't writable at load time. */
#ifndef WEIGHTS_IN_FLASH
#  ifndef WEIGHTS_DEST_ADDR
#    define WEIGHTS_DEST_ADDR  0x342E0000UL
#  endif
    {
        extern const unsigned char _binary_weights_bin_start[];
        extern const unsigned char _binary_weights_bin_end[];
        const uint32_t *s = (const uint32_t *)_binary_weights_bin_start;
        volatile uint32_t *d = (volatile uint32_t *)WEIGHTS_DEST_ADDR;
        uint32_t bytes = (uint32_t)(_binary_weights_bin_end - _binary_weights_bin_start);
        uint32_t n = (bytes + 3) / 4;
        for (uint32_t i = 0; i < n; i++) d[i] = s[i];
    }
    dbg("C:wcopy\n");
#else
    dbg("C:flash-mapped\n");   /* weights are memory-mapped from external flash */
#endif

    kws_benchmark_run();   /* never returns */
    while (1);
}
