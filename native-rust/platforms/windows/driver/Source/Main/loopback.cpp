/*++

Module Name:

    loopback.cpp

Abstract:

    Ring buffer from the Voice Changer speaker to the Voice Changer microphone.
    Positions are total bytes since the driver loaded, so they never wrap in
    practice (2^64 bytes is millions of years at 384 KB/s).

--*/
#include "definitions.h"
#include "loopback.h"

#pragma code_seg()

// Non-paged: the streams call in at DISPATCH_LEVEL from their timer.
static BYTE       g_Ring[LOOPBACK_RING_BYTES];
static ULONGLONG  g_Written = 0;
// A zeroed KSPIN_LOCK is an initialised one.
static KSPIN_LOCK g_RingLock = 0;

VOID LoopbackAlignWriter()
{
    KIRQL irql;
    KeAcquireSpinLock(&g_RingLock, &irql);
    ULONGLONG gap = (LOOPBACK_BLOCK_ALIGN - (g_Written % LOOPBACK_BLOCK_ALIGN)) % LOOPBACK_BLOCK_ALIGN;
    for (ULONGLONG i = 0; i < gap; i++)
    {
        g_Ring[(g_Written + i) % LOOPBACK_RING_BYTES] = 0;
    }
    g_Written += gap;
    KeReleaseSpinLock(&g_RingLock, irql);
}

VOID LoopbackWrite(_In_reads_bytes_(Length) const BYTE* Data, _In_ ULONG Length)
{
    KIRQL irql;
    KeAcquireSpinLock(&g_RingLock, &irql);
    // Only the newest second fits.
    if (Length > LOOPBACK_RING_BYTES)
    {
        g_Written += Length - LOOPBACK_RING_BYTES;
        Data += Length - LOOPBACK_RING_BYTES;
        Length = LOOPBACK_RING_BYTES;
    }
    ULONG offset = (ULONG)(g_Written % LOOPBACK_RING_BYTES);
    ULONG first = min(Length, LOOPBACK_RING_BYTES - offset);
    RtlCopyMemory(g_Ring + offset, Data, first);
    RtlCopyMemory(g_Ring, Data + first, Length - first);
    g_Written += Length;
    KeReleaseSpinLock(&g_RingLock, irql);
}

VOID LoopbackRead(
    _Out_writes_bytes_(Length) BYTE* Data,
    _In_ ULONG Length,
    _In_ ULONGLONG Position,
    _Inout_ LONGLONG* Delta,
    _Inout_ BOOLEAN* Synced)
{
    KIRQL irql;
    KeAcquireSpinLock(&g_RingLock, &irql);
    LONGLONG written = (LONGLONG)g_Written;
    LONGLONG start = (LONGLONG)Position + *Delta;
    LONGLONG lag = written - start;
    if (!*Synced || lag > LOOPBACK_MAX_LAG_BYTES || lag < -LOOPBACK_MAX_LAG_BYTES)
    {
        // Place the read position TARGET_LAG behind the speaker. Delta stays a whole
        // number of frames, so microphone frames line up with speaker frames.
        LONGLONG target = written - LOOPBACK_TARGET_LAG_BYTES;
        LONGLONG delta = target - (LONGLONG)Position;
        LONGLONG remainder = delta % LOOPBACK_BLOCK_ALIGN;
        if (remainder < 0)
        {
            remainder += LOOPBACK_BLOCK_ALIGN;
        }
        *Delta = delta - remainder;
        *Synced = TRUE;
        start = (LONGLONG)Position + *Delta;
    }
    LONGLONG oldest = written - LOOPBACK_RING_BYTES;
    for (ULONG i = 0; i < Length; i++)
    {
        LONGLONG at = start + i;
        Data[i] = (at >= 0 && at >= oldest && at < written)
            ? g_Ring[(ULONGLONG)at % LOOPBACK_RING_BYTES]
            : 0;
    }
    KeReleaseSpinLock(&g_RingLock, irql);
}
