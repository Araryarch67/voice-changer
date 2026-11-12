/*++

Module Name:

    loopback.h

Abstract:

    Voice Changer: what is played to the "Voice Changer Input" speaker comes out of
    the "Voice Changer Mic" microphone. Both pins accept only one format (48 kHz,
    stereo, 32-bit PCM), so bytes are copied unchanged through one shared ring.

--*/
#ifndef _VOICECHANGER_LOOPBACK_H_
#define _VOICECHANGER_LOOPBACK_H_

// The only format both pins accept.
#define LOOPBACK_SAMPLE_RATE        48000
#define LOOPBACK_CHANNELS           2
#define LOOPBACK_BITS_PER_SAMPLE    32
#define LOOPBACK_BLOCK_ALIGN        (LOOPBACK_CHANNELS * LOOPBACK_BITS_PER_SAMPLE / 8)
#define LOOPBACK_BYTES_PER_SECOND   (LOOPBACK_SAMPLE_RATE * LOOPBACK_BLOCK_ALIGN)

// One second of audio.
#define LOOPBACK_RING_BYTES         LOOPBACK_BYTES_PER_SECOND
// The microphone reads this far behind the speaker (20 ms)...
#define LOOPBACK_TARGET_LAG_BYTES   (LOOPBACK_BYTES_PER_SECOND / 50)
// ...and catches up when it falls further behind than 200 ms, or runs ahead.
#define LOOPBACK_MAX_LAG_BYTES      (LOOPBACK_BYTES_PER_SECOND / 5)

// Starts the speaker's data on a frame boundary; call when a render stream runs.
VOID LoopbackAlignWriter();

// Appends what the speaker played.
VOID LoopbackWrite(_In_reads_bytes_(Length) const BYTE* Data, _In_ ULONG Length);

// Fills the microphone's buffer for its stream bytes [Position, Position + Length).
// Delta and Synced belong to the capture stream and keep its place in the ring;
// Synced is FALSE when the stream starts. Missing audio is silence.
VOID LoopbackRead(
    _Out_writes_bytes_(Length) BYTE* Data,
    _In_ ULONG Length,
    _In_ ULONGLONG Position,
    _Inout_ LONGLONG* Delta,
    _Inout_ BOOLEAN* Synced);

#endif // _VOICECHANGER_LOOPBACK_H_
