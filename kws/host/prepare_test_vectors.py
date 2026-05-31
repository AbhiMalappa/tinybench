"""Build the cross-board test vector pack.

Loads the cached MFCC features for the Speech Commands test split, applies the
same normalization the training pipeline used, then quantizes to INT8 with the
input scale/zero_point baked into dscnn_int8.tflite. The result is the exact
490-byte stream every board will receive over UART, plus ground-truth labels
and a reference prediction from the Python TFLite interpreter.

Outputs:
    test_vectors/test_vectors_int8.npy        (N, 490) int8
    test_vectors/test_labels.npy              (N,)     int64
    test_vectors/tflite_reference_preds.npy   (N,)     int64
    test_vectors/test_metadata.json           manifest with hashes + quant params
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mfcc-cache', default='./mfcc_cache/mfcc_testing.pt')
    parser.add_argument('--stats', default='./checkpoints/mfcc_stats.pt',
                        help='Falls back to ds_cnn_checkpoint/mfcc_stats.pt if missing.')
    parser.add_argument('--tflite', default='./to_rohini/dscnn_int8.tflite')
    parser.add_argument('--config', default='./kws/mfcc_config.json')
    parser.add_argument('--out-dir', default='./kws/host/test_vectors')
    args = parser.parse_args()

    if not os.path.exists(args.stats):
        alt = './ds_cnn_checkpoint/mfcc_stats.pt'
        if os.path.exists(alt):
            args.stats = alt
        else:
            sys.exit(f"mfcc_stats.pt not found at {args.stats} or {alt}")

    with open(args.config) as f:
        mfcc_config = json.load(f)
    n_frames = mfcc_config['n_frames']
    n_mfcc = mfcc_config['n_mfcc']
    n_classes = mfcc_config['n_classes']

    print(f"Loading MFCC cache: {args.mfcc_cache}")
    cache = torch.load(args.mfcc_cache, weights_only=True)
    feats = cache['features']  # (N, n_mfcc, n_frames) float32
    labels = cache['labels'].numpy().astype(np.int64)
    n_samples = feats.shape[0]
    print(f"  {n_samples} samples, feature shape {tuple(feats.shape[1:])}")

    print(f"Loading normalization stats: {args.stats}")
    stats = torch.load(args.stats, weights_only=True)
    mean = stats['mean']
    std = stats['std']

    feats_norm = (feats - mean) / (std + 1e-8)
    feats_norm = feats_norm.permute(0, 2, 1).unsqueeze(-1).contiguous().numpy()
    assert feats_norm.shape == (n_samples, n_frames, n_mfcc, 1), feats_norm.shape

    print(f"Loading TFLite model for quantization params: {args.tflite}")
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']
    print(f"  input  dtype={in_det['dtype'].__name__:>6}  scale={in_scale:.6f}  zp={in_zp}")
    print(f"  output dtype={out_det['dtype'].__name__:>6}  scale={out_scale:.6f}  zp={out_zp}")
    assert in_det['dtype'] == np.int8, "expected INT8 input"
    assert tuple(in_det['shape']) == (1, n_frames, n_mfcc, 1), in_det['shape']

    print(f"Quantizing {n_samples} feature tensors to INT8...")
    q = np.clip(np.round(feats_norm / in_scale + in_zp), -128, 127).astype(np.int8)
    vectors = q.reshape(n_samples, n_frames * n_mfcc)
    assert vectors.shape == (n_samples, 490)

    print("Running Python TFLite inference for reference predictions...")
    ref_preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        interpreter.set_tensor(in_det['index'], q[i:i+1])
        interpreter.invoke()
        ref_preds[i] = int(np.argmax(interpreter.get_tensor(out_det['index'])[0]))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_samples}")

    tf_acc = float((ref_preds == labels).mean())
    print(f"Python TFLite reference accuracy: {tf_acc*100:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    vec_path = os.path.join(args.out_dir, 'test_vectors_int8.npy')
    lbl_path = os.path.join(args.out_dir, 'test_labels.npy')
    ref_path = os.path.join(args.out_dir, 'tflite_reference_preds.npy')
    meta_path = os.path.join(args.out_dir, 'test_metadata.json')

    np.save(vec_path, vectors)
    np.save(lbl_path, labels)
    np.save(ref_path, ref_preds)

    metadata = {
        'n_samples': int(n_samples),
        'n_frames': int(n_frames),
        'n_mfcc': int(n_mfcc),
        'n_classes': int(n_classes),
        'input_bytes_per_sample': int(n_frames * n_mfcc),
        'input_scale': float(in_scale),
        'input_zero_point': int(in_zp),
        'output_scale': float(out_scale),
        'output_zero_point': int(out_zp),
        'tflite_reference_accuracy': tf_acc,
        'sources': {
            'tflite_model':      {'path': args.tflite,     'sha256': sha256_file(args.tflite)},
            'mfcc_cache':        {'path': args.mfcc_cache, 'sha256': sha256_file(args.mfcc_cache)},
            'mfcc_stats':        {'path': args.stats,      'sha256': sha256_file(args.stats)},
            'mfcc_config':       {'path': args.config,     'sha256': sha256_file(args.config)},
        },
        'outputs': {
            'test_vectors_int8.npy':     {'sha256': sha256_file(vec_path),
                                          'shape': list(vectors.shape), 'dtype': 'int8'},
            'test_labels.npy':           {'sha256': sha256_file(lbl_path),
                                          'shape': list(labels.shape),  'dtype': 'int64'},
            'tflite_reference_preds.npy':{'sha256': sha256_file(ref_path),
                                          'shape': list(ref_preds.shape), 'dtype': 'int64'},
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote:")
    print(f"  {vec_path}  ({os.path.getsize(vec_path)/1024:.0f} KB)")
    print(f"  {lbl_path}  ({os.path.getsize(lbl_path)/1024:.0f} KB)")
    print(f"  {ref_path}  ({os.path.getsize(ref_path)/1024:.0f} KB)")
    print(f"  {meta_path}")


if __name__ == '__main__':
    main()
