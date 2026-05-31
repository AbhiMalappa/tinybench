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

    print(f"Opening serial: {args.port} @ {args.baud}")
    s = serial.Serial(args.port, args.baud, timeout=0.05)
    time.sleep(1.5)
    s.reset_input_buffer()

    print("Waiting for boot...")
    boot = wait_for_boot(s)
    if boot is not None:
        if boot.get('input_bytes') != 490:
            sys.exit(f"firmware input_bytes={boot.get('input_bytes')} mismatch")
        if boot.get('output_bytes') != 35:
            sys.exit(f"firmware output_bytes={boot.get('output_bytes')} mismatch")
        board_tag = args.board_tag or boot.get('board', 'unknown')
    else:
        boot = {}
        board_tag = args.board_tag or 'unknown'

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.results_dir, f'{board_tag}_dscnn_{ts}.jsonl')
    log_f = open(log_path, 'w')
    log_f.write(json.dumps({'event': 'boot', 'data': boot,
                            'metadata': metadata, 'n_samples': n}) + '\n')

    correct = 0
    agree = 0
    failures = 0
    latencies_ms = []
    t_start = time.time()

    for i in range(n):
        try:
            r = run_one(s, vectors[i].tobytes())
        except Exception as e:
            failures += 1
            print(f"  [{i+1}/{n}] FAILED: {e}")
            log_f.write(json.dumps({'i': i, 'error': str(e)}) + '\n')
            continue

        if 'error' in r:
            failures += 1
            print(f"  [{i+1}/{n}] device error: {r}")
            log_f.write(json.dumps({'i': i, 'response': r}) + '\n')
            continue

        pred = r['class']
        if pred == int(labels[i]):
            correct += 1
        if pred == int(ref_preds[i]):
            agree += 1
        latencies_ms.append(float(r['latency_ms']))
        log_f.write(json.dumps({'i': i, 'label': int(labels[i]),
                                'ref_pred': int(ref_preds[i]), 'response': r}) + '\n')

        if (i + 1) % 10 == 0 or i < 5 or i + 1 == n:
            acc = correct / (i + 1 - failures) * 100 if (i + 1 - failures) > 0 else 0.0
            print(f"  [{i+1}/{n}] pred={pred:2d} label={int(labels[i]):2d} "
                  f"tflite={int(ref_preds[i]):2d}  acc={acc:.1f}%  "
                  f"latency={r['latency_ms']:.2f}ms")

    log_f.close()
    elapsed = time.time() - t_start

    ok = n - failures
    if ok == 0:
        print("\nNo successful runs.")
        sys.exit(2)

    lat = np.array(latencies_ms)
    summary = {
        'board': board_tag,
        'n_attempted': n,
        'n_ok': ok,
        'n_failed': failures,
        'mcu_accuracy': correct / ok,
        'tflite_reference_accuracy_subset': float((ref_preds[:n] == labels[:n]).mean()),
        'mcu_tflite_agreement': agree / ok,
        'latency_ms': {
            'median': float(np.median(lat)),
            'p50': float(np.percentile(lat, 50)),
            'p95': float(np.percentile(lat, 95)),
            'p99': float(np.percentile(lat, 99)),
            'min': float(lat.min()),
            'max': float(lat.max()),
        },
        'arena_used_kb': boot.get('arena_used', 0) / 1024.0,
        'wall_clock_s': elapsed,
        'log_path': log_path,
    }
    summary_path = log_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Board:        {summary['board']}")
    print(f"Samples:      {ok}/{n}  ({failures} failed)")
    print(f"MCU accuracy:                 {summary['mcu_accuracy']*100:.2f}%")
    print(f"TFLite ref accuracy (subset): {summary['tflite_reference_accuracy_subset']*100:.2f}%")
    print(f"MCU↔TFLite agreement:         {summary['mcu_tflite_agreement']*100:.2f}%  ← firmware-correctness signal")
    print(f"Latency median:               {summary['latency_ms']['median']:.3f} ms")
    print(f"Latency p95 / p99:            {summary['latency_ms']['p95']:.3f} / {summary['latency_ms']['p99']:.3f} ms")
    print(f"Peak arena RAM:               {summary['arena_used_kb']:.1f} KB")
    print(f"Wall clock:                   {elapsed:.1f} s  ({elapsed/ok:.2f} s/sample)")
    print(f"Log:                          {log_path}")
    print(f"Summary:                      {summary_path}")


if __name__ == '__main__':
    main()
