"""Build the cross-board test vector pack.

Generates per-model, per-backend test vectors and reference predictions.
Run once per model × backend combination before benchmarking.

Usage:
    # TFLite backend (Arduino Nano 33 BLE, ESP32-S3)
    python kws/host/prepare_test_vectors.py --model dscnn    --backend tflite
    python kws/host/prepare_test_vectors.py --model tcresnet --backend tflite

    # ONNX backend (STM32N6570-DK via Cube.AI)
    python kws/host/prepare_test_vectors.py --model dscnn    --backend onnx
    python kws/host/prepare_test_vectors.py --model tcresnet --backend onnx

Outputs (all in --out-dir):
    test_labels.npy                          (N,) int64        shared — same every run
    test_vectors_int8_{model}.npy            (N, 490) int8     TFLite backend only
    test_vectors_float32.npy                 (N, 490) float32  ONNX backend (Cube.AI takes float input)
    ref_preds_{model}_{backend}.npy          (N,) int64
    test_metadata_{model}_{backend}.json
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def run_tflite(tflite_path, feats_norm, n_frames, n_mfcc, n_samples):
    """Quantize features with TFLite model's input scale/zp and run reference inference."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    print(f"  input  dtype={in_det['dtype'].__name__}  scale={in_scale:.6f}  zp={in_zp}")
    print(f"  output dtype={out_det['dtype'].__name__}  scale={out_scale:.6f}  zp={out_zp}")
    assert in_det['dtype'] == np.int8, "expected INT8 TFLite model"

    model_input_shape = tuple(in_det['shape'])
    assert int(np.prod(model_input_shape)) == n_frames * n_mfcc, \
        f"model input {model_input_shape} ({np.prod(model_input_shape)} elements) != {n_frames * n_mfcc}"

    # feats_norm: (N, n_frames, n_mfcc, 1) NHWC
    feats_nhwc = feats_norm.permute(0, 2, 1).unsqueeze(-1).contiguous().numpy()
    vectors_int8 = np.clip(
        np.round(feats_nhwc / in_scale + in_zp), -128, 127
    ).astype(np.int8).reshape(n_samples, n_frames * n_mfcc)

    print(f"  Running TFLite inference on {n_samples} samples...")
    ref_preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        sample = vectors_int8[i].reshape(model_input_shape).astype(np.int8)
        interp.set_tensor(in_det['index'], sample)
        interp.invoke()
        ref_preds[i] = int(np.argmax(interp.get_tensor(out_det['index'])[0]))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_samples}")

    return vectors_int8, ref_preds, {
        'input_scale': float(in_scale),
        'input_zero_point': int(in_zp),
        'output_scale': float(out_scale),
        'output_zero_point': int(out_zp),
        'input_shape': list(model_input_shape),
    }


def run_onnx(onnx_path, feats_norm, n_frames, n_mfcc, n_samples):
    """Run ONNX reference inference on float32 features (Cube.AI accepts float input)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape  # (batch, 1, n_frames, n_mfcc)
    print(f"  ONNX input: {input_name}  shape={input_shape}")

    # ONNX model expects (batch=1, channel=1, n_frames, n_mfcc) — NCHW
    # feats_norm raw: (N, n_mfcc, n_frames); permute → (N, n_frames, n_mfcc); unsqueeze → (N, 1, n_frames, n_mfcc)
    feats_nchw = feats_norm.permute(0, 2, 1).unsqueeze(1).contiguous().numpy()  # (N, 1, 49, 10)

    print(f"  Running ONNX inference on {n_samples} samples...")
    ref_preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        logits = sess.run(None, {input_name: feats_nchw[i:i+1]})[0]
        ref_preds[i] = int(np.argmax(logits[0]))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_samples}")

    # Float32 vectors: flat (N, 490) — firmware reshapes to its own input format
    vectors_float32 = feats_norm.permute(0, 2, 1).contiguous().numpy().reshape(n_samples, n_frames * n_mfcc)

    return vectors_float32, ref_preds, {
        'input_name': input_name,
        'input_shape_nchw': [1, 1, n_frames, n_mfcc],
        'input_dtype': 'float32',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['dscnn', 'tcresnet', 'gru'],
                        help='Model name — determines default model path and output filenames')
    parser.add_argument('--backend', default='tflite', choices=['tflite', 'onnx'],
                        help='Inference backend: tflite (Arduino/ESP32) or onnx (STM32/Cube.AI)')
    parser.add_argument('--model-path', default=None,
                        help='Override model file path (default: checkpoints/{model}_int8.{tflite|onnx})')
    parser.add_argument('--mfcc-cache', default='./mfcc_cache/mfcc_testing.pt')
    parser.add_argument('--stats', default='./checkpoints/mfcc_stats.pt')
    parser.add_argument('--config', default='./kws/mfcc_config.json')
    parser.add_argument('--out-dir', default='./kws/host/test_vectors')
    args = parser.parse_args()

    # Resolve model file path
    if args.model_path:
        model_path = args.model_path
    else:
        ext = 'tflite' if args.backend == 'tflite' else 'onnx'
        model_path = f'./checkpoints/{args.model}_int8.{ext}'

    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}")

    if not os.path.exists(args.stats):
        alt = './ds_cnn_checkpoint/mfcc_stats.pt'
        if os.path.exists(alt):
            args.stats = alt
        else:
            sys.exit(f"mfcc_stats.pt not found at {args.stats} or {alt}")

    with open(args.config) as f:
        mfcc_config = json.load(f)
    n_frames  = mfcc_config['n_frames']
    n_mfcc    = mfcc_config['n_mfcc']
    n_classes = mfcc_config['n_classes']

    print(f"Model:   {args.model}  backend={args.backend}  path={model_path}")
    print(f"Loading MFCC cache: {args.mfcc_cache}")
    cache = torch.load(args.mfcc_cache, weights_only=True)
    feats  = cache['features']                          # (N, n_mfcc, n_frames) float32
    labels = cache['labels'].numpy().astype(np.int64)
    n_samples = feats.shape[0]
    print(f"  {n_samples} samples, raw feature shape {tuple(feats.shape[1:])}")

    print(f"Loading normalization stats: {args.stats}")
    stats = torch.load(args.stats, weights_only=True)
    feats_norm = (feats - stats['mean']) / (stats['std'] + 1e-8)  # (N, n_mfcc, n_frames)

    print(f"Loading {args.backend.upper()} model: {model_path}")
    if args.backend == 'tflite':
        vectors, ref_preds, quant_params = run_tflite(
            model_path, feats_norm, n_frames, n_mfcc, n_samples)
        vec_dtype = 'int8'
    else:
        vectors, ref_preds, quant_params = run_onnx(
            model_path, feats_norm, n_frames, n_mfcc, n_samples)
        vec_dtype = 'float32'

    ref_acc = float((ref_preds == labels).mean())
    print(f"Reference accuracy ({args.backend}): {ref_acc*100:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)

    # Shared: labels (model/backend-independent)
    lbl_path = os.path.join(args.out_dir, 'test_labels.npy')
    np.save(lbl_path, labels)

    # Per-model INT8 vectors (TFLite) or shared float32 vectors (ONNX)
    if args.backend == 'tflite':
        vec_path = os.path.join(args.out_dir, f'test_vectors_int8_{args.model}.npy')
    else:
        vec_path = os.path.join(args.out_dir, 'test_vectors_float32.npy')
    np.save(vec_path, vectors)

    # Per-model, per-backend reference predictions
    ref_path = os.path.join(args.out_dir, f'ref_preds_{args.model}_{args.backend}.npy')
    np.save(ref_path, ref_preds)

    metadata = {
        'model': args.model,
        'backend': args.backend,
        'n_samples': int(n_samples),
        'n_frames': int(n_frames),
        'n_mfcc': int(n_mfcc),
        'n_classes': int(n_classes),
        'input_bytes_per_sample': int(vectors.shape[1] * vectors.itemsize),
        'reference_accuracy': ref_acc,
        'quant_params': quant_params,
        'sources': {
            'model':      {'path': model_path,     'sha256': sha256_file(model_path)},
            'mfcc_cache': {'path': args.mfcc_cache,'sha256': sha256_file(args.mfcc_cache)},
            'mfcc_stats': {'path': args.stats,     'sha256': sha256_file(args.stats)},
            'mfcc_config':{'path': args.config,    'sha256': sha256_file(args.config)},
        },
        'outputs': {
            os.path.basename(vec_path): {
                'sha256': sha256_file(vec_path),
                'shape': list(vectors.shape), 'dtype': vec_dtype},
            'test_labels.npy': {
                'sha256': sha256_file(lbl_path),
                'shape': list(labels.shape), 'dtype': 'int64'},
            os.path.basename(ref_path): {
                'sha256': sha256_file(ref_path),
                'shape': list(ref_preds.shape), 'dtype': 'int64'},
        },
    }
    meta_path = os.path.join(args.out_dir, f'test_metadata_{args.model}_{args.backend}.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote:")
    print(f"  {lbl_path}  ({os.path.getsize(lbl_path)/1024:.0f} KB)  [shared]")
    print(f"  {vec_path}  ({os.path.getsize(vec_path)/1024:.0f} KB)")
    print(f"  {ref_path}  ({os.path.getsize(ref_path)/1024:.0f} KB)")
    print(f"  {meta_path}")


if __name__ == '__main__':
    main()
