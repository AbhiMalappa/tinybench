"""Drive a TinyBench-KWS firmware over serial and collect benchmark metrics.

Loads the precomputed test vector pack, sends raw INT8 bytes one tensor at a
time, parses the JSON response, and aggregates accuracy and latency. The host
code is identical for every board; only the firmware behind the serial port
differs. See protocol.md for the wire contract.

Usage:
    python benchmark_serial.py --port /dev/cu.usbmodem14101 --n-samples 20
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import serial


def read_line(s, timeout_s=10.0):
    deadline = time.time() + timeout_s
    buf = bytearray()
    while time.time() < deadline:
        b = s.read(1)
        if not b:
            continue
        if b == b'\n':
            return buf.decode('utf-8', errors='replace').rstrip('\r')
        buf.extend(b)
    raise TimeoutError(f"line read timed out after {timeout_s}s, buffer={buf!r}")


def wait_for_boot(s, timeout_s=4.0):
    """Try to capture a boot JSON. If none arrives (device already mid-loop),
    return None and let the caller proceed without boot validation."""
    deadline = time.time() + timeout_s
    boot = None
    while time.time() < deadline:
        try:
            line = read_line(s, timeout_s=max(0.1, deadline - time.time()))
        except TimeoutError:
            break
        if not line:
            continue
        if line == 'READY':
            if boot is None:
                print("  (got READY without preceding BOOT — device was already running)")
            return boot
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            print(f"  (ignored non-JSON pre-boot line: {line!r})")
            continue
        if j.get('event') == 'boot':
            boot = j
            print(f"BOOT: {boot}")
        elif 'error' in j:
            raise RuntimeError(f"device error during boot: {j}")
    if boot is None:
        print("  (no boot message seen — assuming device was already running)")
    return boot


def write_chunked(s, data, chunk_size=32, delay_s=0.050):
    """Mbed Nano 33 BLE USB CDC drops bytes on bulk writes. Trickle it in."""
    for off in range(0, len(data), chunk_size):
        s.write(data[off:off + chunk_size])
        s.flush()
        time.sleep(delay_s)


def run_one(s, payload_bytes, response_timeout_s=30.0):
    assert len(payload_bytes) == 490
    write_chunked(s, payload_bytes)
    # The firmware may emit diagnostic lines before the result; the result is
    # whichever line parses to a dict containing a 'class' or 'error' key.
    while True:
        line = read_line(s, timeout_s=response_timeout_s)
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'class' in j or 'error' in j:
            ready_line = read_line(s, timeout_s=response_timeout_s)
            if ready_line != 'READY':
                raise RuntimeError(f"expected READY after result, got {ready_line!r}")
            return j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--vectors-dir', default='./kws/host/test_vectors')
    parser.add_argument('--n-samples', type=int, default=20,
                        help='How many test vectors to send (use 0 for all)')
    parser.add_argument('--results-dir', default='./kws/host/results')
    parser.add_argument('--board-tag', default=None,
                        help='Override device board id used in output filename')
    parser.add_argument('--resume-from', default=None,
                        help='Path to existing JSONL log. Appends to that file and skips '
                             'clip indices already completed. Pair with --reflash-build-dir '
                             'for safe resume after mid-clip interruption.')
    parser.add_argument('--reflash-build-dir', default=None,
                        help='Run `arduino-cli upload --input-dir <dir>` before opening '
                             'serial. Guarantees a clean device state — critical for resume '
                             'safety and avoiding stale-state failures from prior runs.')
    parser.add_argument('--fqbn', default='arduino:mbed_nano:nano33ble',
                        help='Board FQBN used when --reflash-build-dir is set.')
    args = parser.parse_args()

    vec_path = os.path.join(args.vectors_dir, 'test_vectors_int8.npy')
    lbl_path = os.path.join(args.vectors_dir, 'test_labels.npy')
    ref_path = os.path.join(args.vectors_dir, 'tflite_reference_preds.npy')
    meta_path = os.path.join(args.vectors_dir, 'test_metadata.json')

    vectors = np.load(vec_path)
    labels = np.load(lbl_path)
    ref_preds = np.load(ref_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    assert vectors.dtype == np.int8 and vectors.shape[1] == 490, vectors.shape
    n_total = vectors.shape[0]
    n = args.n_samples if args.n_samples > 0 else n_total
    n = min(n, n_total)
    print(f"Vectors:   {vec_path}  ({n_total} total, running {n})")
    print(f"Reference TFLite accuracy: {metadata['tflite_reference_accuracy']*100:.2f}%")

    if args.reflash_build_dir:
        print(f"Re-flashing {args.fqbn} from {args.reflash_build_dir}...")
        subprocess.check_call([
            'arduino-cli', 'upload',
            '--fqbn', args.fqbn,
            '--port', args.port,
            '--input-dir', args.reflash_build_dir,
        ])
        print("Flash complete; sleeping 4 s for USB re-enumeration.")
        time.sleep(4.0)

    print(f"Opening serial: {args.port} @ {args.baud}")
    s = serial.Serial(args.port, args.baud, timeout=0.05)
    s.reset_input_buffer()
    time.sleep(1.5)

    print("Waiting for boot...")
    boot = wait_for_boot(s)
    if boot is not None:
        if boot.get('input_bytes') != 490:
            sys.exit(f"firmware input_bytes={boot.get('input_bytes')} mismatch")
        if boot.get('output_bytes') != 35:
            sys.exit(f"firmware output_bytes={boot.get('output_bytes')} mismatch")
        board_tag = args.board_tag or boot.get('board', 'unknown')
        model_tag = boot.get('model', 'unknown')
    else:
        boot = {}
        board_tag = args.board_tag or 'unknown'
        model_tag = 'unknown'

    completed_indices = set()
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            sys.exit(f"--resume-from path does not exist: {args.resume_from}")
        with open(args.resume_from) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(j, dict) and 'i' in j:
                    r = j.get('response')
                    if isinstance(r, dict) and 'class' in r:
                        completed_indices.add(j['i'])
        log_path = args.resume_from
        log_f = open(log_path, 'a')
        log_f.write(json.dumps({'event': 'resume', 'ts': datetime.now().isoformat(),
                                'already_completed': len(completed_indices)}) + '\n')
        print(f"Resuming: {len(completed_indices)} clips already completed in {log_path}")
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(args.results_dir, f'{board_tag}_{model_tag}_{ts}.jsonl')
        log_f = open(log_path, 'w')
        log_f.write(json.dumps({'event': 'boot', 'data': boot,
                                'metadata': metadata, 'n_samples': n}) + '\n')

    session_failures = 0
    session_ran = 0
    t_start = time.time()

    for i in range(n):
        if i in completed_indices:
            continue
        try:
            r = run_one(s, vectors[i].tobytes())
        except Exception as e:
            session_failures += 1
            print(f"  [{i+1}/{n}] FAILED: {e}")
            log_f.write(json.dumps({'i': i, 'error': str(e)}) + '\n')
            log_f.flush()
            continue

        if 'error' in r:
            session_failures += 1
            print(f"  [{i+1}/{n}] device error: {r}")
            log_f.write(json.dumps({'i': i, 'response': r}) + '\n')
            log_f.flush()
            continue

        session_ran += 1
        log_f.write(json.dumps({'i': i, 'label': int(labels[i]),
                                'ref_pred': int(ref_preds[i]), 'response': r}) + '\n')
        log_f.flush()

        if session_ran % 10 == 0 or session_ran <= 5 or i + 1 == n:
            print(f"  [{i+1}/{n}] pred={r['class']:2d} label={int(labels[i]):2d} "
                  f"tflite={int(ref_preds[i]):2d}  latency={r['latency_ms']:.2f}ms")

    log_f.close()
    elapsed = time.time() - t_start

    # Cumulative metrics: re-read the full log so resume + fresh runs report uniformly.
    cum_correct = cum_agree = cum_failed = 0
    cum_latencies = []
    completed_total = set()
    with open(log_path) as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(j, dict) or 'i' not in j:
                continue
            r = j.get('response')
            if isinstance(r, dict) and 'class' in r:
                completed_total.add(j['i'])
                pred = r['class']
                i_idx = j['i']
                if pred == int(labels[i_idx]):
                    cum_correct += 1
                if pred == int(ref_preds[i_idx]):
                    cum_agree += 1
                cum_latencies.append(float(r['latency_ms']))
            else:
                cum_failed += 1

    n_cum = len(completed_total)
    if n_cum == 0:
        print("\nNo successful runs.")
        sys.exit(2)

    lat = np.array(cum_latencies)
    summary = {
        'board': board_tag,
        'n_target': n,
        'n_completed_cumulative': n_cum,
        'n_failed_cumulative': cum_failed,
        'session_ran': session_ran,
        'session_failed': session_failures,
        'mcu_accuracy': cum_correct / n_cum,
        'mcu_tflite_agreement': cum_agree / n_cum,
        'latency_ms': {
            'median': float(np.median(lat)),
            'p50': float(np.percentile(lat, 50)),
            'p95': float(np.percentile(lat, 95)),
            'p99': float(np.percentile(lat, 99)),
            'min': float(lat.min()),
            'max': float(lat.max()),
        },
        'arena_used_kb': boot.get('arena_used', 0) / 1024.0,
        'session_wall_clock_s': elapsed,
        'log_path': log_path,
    }
    summary_path = log_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Board:                       {summary['board']}")
    print(f"Cumulative completed:        {n_cum}/{n}  ({cum_failed} failed)")
    print(f"This session:                {session_ran} new clips ({session_failures} failed)")
    print(f"MCU accuracy:                {summary['mcu_accuracy']*100:.2f}%")
    print(f"MCU↔TFLite agreement:        {summary['mcu_tflite_agreement']*100:.2f}%  ← firmware-correctness signal")
    print(f"Latency median:              {summary['latency_ms']['median']:.3f} ms")
    print(f"Latency p95 / p99:           {summary['latency_ms']['p95']:.3f} / {summary['latency_ms']['p99']:.3f} ms")
    print(f"Peak arena RAM:              {summary['arena_used_kb']:.1f} KB")
    print(f"Session wall clock:          {elapsed:.1f} s")
    print(f"Log:                         {log_path}")
    print(f"Summary:                     {summary_path}")


if __name__ == '__main__':
    main()
