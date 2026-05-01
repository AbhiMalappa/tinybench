"""Quantize trained KWS model: PyTorch float32 → ONNX → TFLite INT8.

Outputs saved to checkpoints/:
  <model>.onnx           for STM32 Cube.AI (partner workflow)
  <model>_int8.tflite    for TFLite Micro (ESP32-S3, Arduino Nano 33)

Usage (run from tinybench/):
    python kws/quantize.py --model dscnn
    python kws/quantize.py --model tcresnet
    python kws/quantize.py --model gru
"""

import argparse
import json
import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kws.data.dataset import get_dataloaders
from kws.models.dscnn import DSCNN
from kws.models.tcresnet import TCResNet8
from kws.models.gru import GRU48


def load_model(model_name, config, ckpt_path):
    if model_name == 'dscnn':
        model = DSCNN(n_classes=config['n_classes'])
    elif model_name == 'tcresnet':
        model = TCResNet8(n_classes=config['n_classes'], n_mfcc=config['n_mfcc'])
    elif model_name == 'gru':
        model = GRU48(n_classes=config['n_classes'], n_mfcc=config['n_mfcc'])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


@torch.no_grad()
def evaluate_float32(model, loader):
    correct = total = 0
    for feats, labels in loader:
        correct += (model(feats).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def export_onnx(model, onnx_path, n_mfcc):
    dummy = torch.zeros(1, 1, 49, n_mfcc)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['mfcc'],
        output_names=['logits'],
        dynamic_axes={'mfcc': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX exported: {onnx_path}  ({size_kb:.1f} KB)")
    return size_kb


def convert_to_tflite_int8(onnx_path, tflite_path, test_loader, n_calib):
    """ONNX → TF SavedModel (onnx2tf) → TFLite INT8 (per-channel weights)."""
    import tensorflow as tf
    import onnx2tf

    tf_dir = tempfile.mkdtemp()
    print("Converting ONNX → TF SavedModel...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_dir,
        non_verbose=True,
    )

    def representative_dataset():
        count = 0
        for feats, _ in test_loader:
            for i in range(feats.shape[0]):
                if count >= n_calib:
                    return
                yield [feats[i:i+1].numpy().astype(np.float32)]
                count += 1

    print(f"Quantizing to INT8 (calibration samples: {n_calib})...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"TFLite INT8 saved: {tflite_path}  ({size_kb:.1f} KB)")
    return tflite_model, size_kb


def evaluate_tflite_int8(tflite_model, test_loader):
    """Run full test set through TFLite INT8 interpreter and return accuracy."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    correct = total = 0
    for feats, labels in test_loader:
        for i in range(feats.shape[0]):
            x = feats[i:i+1].numpy().astype(np.float32)
            x_int8 = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            interpreter.set_tensor(inp['index'], x_int8)
            interpreter.invoke()
            logits = interpreter.get_tensor(out['index']).astype(np.float32)
            pred = np.argmax((logits - out_zp) * out_scale)
            correct += int(pred == labels[i].item())
            total += 1
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['dscnn', 'tcresnet', 'gru'])
    parser.add_argument('--config', default='./kws/mfcc_config.json')
    parser.add_argument('--data-root', default='./data/speechcommands')
    parser.add_argument('--checkpoints-dir', default='./kws/checkpoints')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-calib', type=int, default=500,
                        help='Calibration samples for INT8 quantization — same value across all models')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    stats_path = os.path.join(args.checkpoints_dir, 'mfcc_stats.pt')
    _, _, test_loader, _ = get_dataloaders(
        args.data_root, args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stats_path=stats_path,
    )

    ckpt = os.path.join(args.checkpoints_dir, f'{args.model}_best.pt')
    model = load_model(args.model, config, ckpt)

    print(f"\n=== {args.model.upper()} ===")
    float32_acc = evaluate_float32(model, test_loader)
    print(f"Float32 accuracy: {float32_acc*100:.2f}%")

    onnx_path = os.path.join(args.checkpoints_dir, f'{args.model}.onnx')
    onnx_kb = export_onnx(model, onnx_path, config['n_mfcc'])

    tflite_path = os.path.join(args.checkpoints_dir, f'{args.model}_int8.tflite')
    tflite_model, tflite_kb = convert_to_tflite_int8(
        onnx_path, tflite_path, test_loader, n_calib=args.n_calib
    )

    print("Evaluating INT8 accuracy on test set...")
    int8_acc = evaluate_tflite_int8(tflite_model, test_loader)
    acc_drop = (float32_acc - int8_acc) * 100

    print(f"\n--- Results: {args.model.upper()} ---")
    print(f"  Float32 accuracy  : {float32_acc*100:.2f}%")
    print(f"  INT8 accuracy     : {int8_acc*100:.2f}%")
    print(f"  Accuracy drop     : {acc_drop:.2f}%")
    print(f"  ONNX size (flash) : {onnx_kb:.1f} KB")
    print(f"  TFLite INT8 size  : {tflite_kb:.1f} KB")


if __name__ == '__main__':
    main()
