/*
 * Minimal newlib-nano syscall stubs for the bare-metal benchmark firmware.
 *
 * Provides __errno (referenced by libm float-math error handling) and a simple
 * _sbrk heap (the X-CUBE-AI runtime may allocate a little). All I/O stubs fail
 * gracefully — the firmware talks to the host via HAL UART directly, not stdio.
 */
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <stddef.h>

/* errno storage + accessor (newlib: errno is (*__errno())) */
static int _errno_storage;
int *__errno(void) { return &_errno_storage; }

/* Heap, grows from _end (linker symbol) toward the stack region. */
extern char _end;            /* set by linker script */
extern char _estack;         /* top of stack */

void *_sbrk(ptrdiff_t incr)
{
    static char *heap = 0;
    char *prev, *next;
    if (heap == 0) heap = &_end;
    prev = heap;
    next = heap + incr;
    /* keep a 16 KB guard below the stack top */
    if (next > (&_estack - 0x4000)) {
        errno = ENOMEM;
        return (void *)-1;
    }
    heap = next;
    return (void *)prev;
}

int  _close(int f)                       { (void)f; return -1; }
int  _fstat(int f, struct stat *st)      { (void)f; st->st_mode = S_IFCHR; return 0; }
int  _isatty(int f)                      { (void)f; return 1; }
int  _lseek(int f, int p, int w)         { (void)f; (void)p; (void)w; return 0; }
int  _read(int f, char *p, int n)        { (void)f; (void)p; (void)n; return 0; }

/* Route printf/stdout to USART1 so LL_ATON runtime diagnostics (LL_ATON_PRINTF)
   are visible during NPU bring-up. huart1 is set up by uart_init() in main. */
#include "stm32n6xx_hal.h"
extern UART_HandleTypeDef huart1;
int  _write(int f, const char *p, int n)
{
    (void)f;
    if (huart1.Instance) HAL_UART_Transmit(&huart1, (uint8_t *)p, n, HAL_MAX_DELAY);
    return n;
}
void _exit(int c)                        { (void)c; while (1); }
int  _kill(int pid, int sig)             { (void)pid; (void)sig; errno = EINVAL; return -1; }
int  _getpid(void)                       { return 1; }
