/*
 * TinyBench-KWS STM32N6570-DK firmware entry point.
 * Hardware: STM32N6570-DK rev C01+, Cortex-M55 @ 800 MHz
 * Protocol: see kws/host/protocol.md
 */
#include <string.h>
#include "stm32n6xx_hal.h"

UART_HandleTypeDef huart1;

void kws_benchmark_run(void);

/* ── Vector table / IRQ setup (must run before HAL_Init) ─────────────────── */
static void board_early_init(void)
{
    __disable_irq();
    SCB->VTOR = 0x34000000;
    __DSB();
    /* Clear all NVIC enables and pending bits left by bootrom */
    memset((uint32_t *)NVIC->ICER, 0xFF, sizeof(NVIC->ICER));
    memset((uint32_t *)NVIC->ICPR, 0xFF, sizeof(NVIC->ICPR));
    __enable_irq();
}

/* ── Power-domain bringup (must precede UART/GPIO use on STM32N6) ──────────── */
static void power_init(void)
{
    /* N6 quirk: force-enable the bus (APB) clocks, otherwise USART1 stays dark */
    RCC->BUSENSR = 0xFFFFFFFF;

    /* Power the analog + I/O supply domains. GPIO banks (incl. GPIOE for USART1
       PE5/PE6) will not drive their pins unless their VddIO domain is enabled. */
    HAL_PWREx_EnableVddA();
    HAL_PWREx_EnableVddIO2();
    HAL_PWREx_EnableVddIO3();
    HAL_PWREx_EnableVddIO4();
    HAL_PWREx_EnableVddIO5();

    __HAL_RCC_PWR_CLK_ENABLE();
    if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK) while (1);
    if (HAL_PWREx_GetSupplyConfig() != PWR_EXTERNAL_SOURCE_SUPPLY) while (1);
}

/* ── SRAM banks + cache-controller unlock ─────────────────────────────────── */
static void memory_init(void)
{
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    __HAL_RCC_CRC_CLK_ENABLE();

    /* Enable AXISRAM2-6 and AXI cache RAM */
    RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN |
                   RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;
    RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN;

    /* Take SRAMs out of shutdown */
    RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;

    /* Allow cache activation (bootrom sets these to 0) */
    MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

/* ── Pure busy-loop delay (needs neither SysTick nor DWT) ─────────────────── */
static void delay_busy(volatile uint32_t iters)
{
    while (iters--) { __NOP(); }
}

/* ── SMPS overdrive for 800 MHz (PF4 = HIGH, DK rev C01+) ───────────────── */
static void smps_overdrive(void)
{
    __HAL_RCC_GPIOF_CLK_ENABLE();
    GPIO_InitTypeDef g = {0};
    g.Pin   = GPIO_PIN_4;
    g.Mode  = GPIO_MODE_OUTPUT_PP;
    g.Pull  = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOF, &g);
    HAL_GPIO_WritePin(GPIOF, GPIO_PIN_4, GPIO_PIN_SET);
    /* Voltage ramp wait. Pure busy-loop — avoids SysTick (HAL_Delay) and DWT,
       both of which can be non-functional this early after the bootrom hand-off. */
    delay_busy(500000);
}

/* ── System clock: PLL1 → 800 MHz CPU ───────────────────────────────────── */
static void clock_config_800mhz(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    /* Supply already configured in power_init(); set overdrive voltage scale for 800 MHz */
    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE0) != HAL_OK) while(1);

    /* Step 1: bring clocks back to HSI so PLL1 can be reconfigured */
    HAL_RCC_GetClockConfig(&clk);
    if ((clk.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
        (clk.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11)) {
        clk.ClockType   = RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK;
        clk.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
        clk.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
        if (HAL_RCC_ClockConfig(&clk) != HAL_OK) while(1);
    }

    /* Step 2: PLL1 — HSI(64 MHz)/8 × 100 = 800 MHz VCO */
    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState       = RCC_HSI_ON;
    osc.HSIDiv         = RCC_HSI_DIV1;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL1.PLLState  = RCC_PLL_ON;
    osc.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
    osc.PLL1.PLLM      = 8;
    osc.PLL1.PLLN      = 100;
    osc.PLL1.PLLFractional = 0;
    osc.PLL1.PLLP1     = 1;
    osc.PLL1.PLLP2     = 1;
    osc.PLL2.PLLState  = RCC_PLL_NONE;
    osc.PLL3.PLLState  = RCC_PLL_NONE;
    osc.PLL4.PLLState  = RCC_PLL_NONE;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) while(1);

    /* Step 3: route PLL1 → CPU @ 800 MHz, AXI @ 400 MHz, AHB @ 200 MHz */
    clk.ClockType = RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_HCLK |
                    RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 |
                    RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_PCLK5 |
                    RCC_CLOCKTYPE_PCLK4;
    clk.CPUCLKSource  = RCC_CPUCLKSOURCE_IC1;
    clk.SYSCLKSource  = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
    clk.AHBCLKDivider = RCC_HCLK_DIV2;
    clk.APB1CLKDivider = RCC_APB1_DIV1;
    clk.APB2CLKDivider = RCC_APB2_DIV1;
    clk.APB4CLKDivider = RCC_APB4_DIV1;
    clk.APB5CLKDivider = RCC_APB5_DIV1;
    clk.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC1Selection.ClockDivider   = 1;   /* CPU = 800 MHz */
    clk.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC2Selection.ClockDivider   = 2;   /* SYSCLK = 400 MHz */
    clk.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC6Selection.ClockDivider   = 4;
    clk.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
    clk.IC11Selection.ClockDivider   = 2;
    if (HAL_RCC_ClockConfig(&clk) != HAL_OK) while(1);
}

/* ── USART1 init: PE5=TX, PE6=RX, 115200 8N1 ────────────────────────────── */
static void uart_init(void)
{
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_USART1_FORCE_RESET();
    __HAL_RCC_USART1_RELEASE_RESET();
    __HAL_RCC_GPIOE_CLK_ENABLE();

    GPIO_InitTypeDef g = {0};
    g.Pin       = GPIO_PIN_5;
    g.Mode      = GPIO_MODE_AF_PP;
    g.Pull      = GPIO_NOPULL;
    g.Speed     = GPIO_SPEED_FREQ_LOW;
    g.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOE, &g);

    g.Pin  = GPIO_PIN_6;
    g.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOE, &g);

    huart1.Instance          = USART1;
    huart1.Init.BaudRate     = 115200;
    huart1.Init.WordLength   = UART_WORDLENGTH_8B;
    huart1.Init.StopBits     = UART_STOPBITS_1;
    huart1.Init.Parity       = UART_PARITY_NONE;
    huart1.Init.Mode         = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl   = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart1) != HAL_OK) while(1);
    /* Disable FIFO mode */
    huart1.Instance->CR1 &= ~USART_CR1_FIFOEN;
}

/* ── MPU: mark a non-cacheable region (required by HAL) ─────────────────── */
static void mpu_config(void)
{
    MPU_Region_InitTypeDef reg  = {0};
    MPU_Attributes_InitTypeDef attr = {0};
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    HAL_MPU_Disable();
    attr.Attributes = INNER_OUTER(MPU_NOT_CACHEABLE);
    attr.Number     = MPU_ATTRIBUTES_NUMBER0;
    HAL_MPU_ConfigMemoryAttributes(&attr);
    /* Region 0: non-cacheable placeholder (zero length disabled) */
    reg.Enable           = MPU_REGION_ENABLE;
    reg.Number           = MPU_REGION_NUMBER0;
    reg.BaseAddress      = 0x34000000;
    reg.LimitAddress     = 0x34000000 + 0x200000 - 1;
    reg.DisableExec      = MPU_INSTRUCTION_ACCESS_ENABLE;
    reg.AccessPermission = MPU_REGION_ALL_RW;
    reg.IsShareable      = MPU_ACCESS_NOT_SHAREABLE;
    reg.AttributesIndex  = MPU_ATTRIBUTES_NUMBER0;
    HAL_MPU_ConfigRegion(&reg);
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
    __set_PRIMASK(primask);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(void)
{
    board_early_init();

    SCB_EnableICache();
    SCB_EnableDCache();

    HAL_Init();

    power_init();

    memory_init();

    /* Fix clock source before reconfiguring (may be on IC1 after bootrom) */
    __HAL_RCC_CPUCLK_CONFIG(RCC_CPUCLKSOURCE_HSI);
    __HAL_RCC_SYSCLK_CONFIG(RCC_SYSCLKSOURCE_HSI);

    /* Enable SMPS overdrive BEFORE going to 800 MHz */
    smps_overdrive();

    clock_config_800mhz();

    mpu_config();

    uart_init();

    kws_benchmark_run();   /* never returns */
    while(1);
}
