/*
 * Minimal SystemInit for the TinyBench KWS benchmark firmware (SRAM debug image).
 *
 * The STM32N6 bootrom has already configured security (SAU) and brought the core
 * up before handing control to this debugger-loaded image, so we do NOT repeat the
 * full FSBL secure setup (which requires the -mcmse secure toolchain and pulls in
 * SAU/SCB_NS symbols). We only do what main() relies on before its own bringup:
 * enable the FPU so floating-point code (the benchmark loop) runs. The vector table
 * is relocated to 0x34000000 explicitly in main()'s board_early_init().
 */
#include "stm32n6xx.h"

uint32_t SystemCoreClock = 64000000U;   /* HSI at reset; updated after clock_config */

void SystemCoreClockUpdate(void) { /* not used; clock set explicitly in main */ }

void SystemInit(void)
{
    /* Enable CP10 and CP11 (FPU) full access */
    SCB->CPACR |= ((3UL << 20U) | (3UL << 22U));
    __DSB();
    __ISB();
}
