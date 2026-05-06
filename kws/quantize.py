"""Quantize trained KWS model: PyTorch float32 → ONNX → INT8 ONNX + TFLite INT8.

Outputs saved to checkpoints/:
  <model>.onnx           float32 ONNX  (baseline / partner inspection)
  <model>_int8.onnx      INT8 ONNX via ONNX Runtime  (STM32 Cube.AI primary path)
  <model>_int8.tflite    TFLite INT8  (TFLite Micro: ESP32-S3, Arduino Nano 33)

Usage (run from tinybench/):
    python kws/quantize.py --model dscnn
    python kws/quantize.py --model tcresnet
    python kws/quantize.py --model gru
    python kws/quantize.py --model dscnn --skip-tflite   # INT8 ONNX only
"""

import argparse
import inspect
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


# ---------------------------------------------------------------------------
# INT8 ONNX via ONNX Runtime static quantization
# ---------------------------------------------------------------------------

def quantize_onnx_int8(onnx_path, int8_onnx_path, test_loader, n_calib):
    """ONNX float32 → ONNX INT8 using ONNX Runtime static quantization.

    Produces a QDQ-format INT8 ONNX directly consumable by STM32 Cube.AI.
    """
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    )
    import onnxruntime as ort

    # Collect calibration samples
    sess_tmp = ort.InferenceSession(onnx_path,
                                    providers=['CPUExecutionProvider'])
    input_name = sess_tmp.get_inputs()[0].name

    class MFCCReader(CalibrationDataReader):
        def __init__(self):
            data = []
            count = 0
            for feats, _ in test_loader:
                for i in range(feats.shape[0]):
                    if count >= n_calib:
                        break
                    data.append({input_name: feats[i:i+1].numpy().astype(np.float32)})
                    count += 1
                if count >= n_calib:
                    break
            self._iter = iter(data)

        def get_next(self):
            return next(self._iter, None)

    # Pre-process adds shape info and makes quantization more accurate
    preprocessed = int8_onnx_path.replace('.onnx', '_prep.onnx')
    try:
        from onnxruntime.quantization import quant_pre_process
        quant_pre_process(onnx_path, preprocessed, skip_optimization=False)
        src = preprocessed
    except Exception:
        src = onnx_path

    print(f"Quantizing ONNX to INT8 (calibration: {n_calib} samples)...")
    quantize_static(
        src,
        int8_onnx_path,
        MFCCReader(),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
    )

    if os.path.exists(preprocessed):
        os.remove(preprocessed)

    size_kb = os.path.getsize(int8_onnx_path) / 1024
    print(f"INT8 ONNX saved: {int8_onnx_path}  ({size_kb:.1f} KB)")
    return size_kb


def evaluate_int8_onnx(int8_onnx_path, test_loader):
    """Run full test set through INT8 ONNX and return accuracy."""
    import onnxruntime as ort

    sess = ort.InferenceSession(int8_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    correct = total = 0
    for feats, labels in test_loader:
        logits = sess.run(None, {input_name: feats.numpy()})[0]
        preds = np.argmax(logits, axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# TFLite INT8 via onnx2tf  (handles multiple onnx2tf API versions)
# ---------------------------------------------------------------------------

def _collect_calib_samples(test_loader, n_calib):
    samples = []
    count = 0
    for feats, _ in test_loader:
        for i in range(feats.shape[0]):
            if count >= n_calib:
                break
            samples.append(feats[i:i+1].numpy().astype(np.float32))
            count += 1
        if count >= n_calib:
            break
    return samples


def convert_to_tflite_int8(onnx_path, tflite_path, test_loader, n_calib):
    """ONNX → TFLite INT8.

    Strategy:
      1. If installed onnx2tf supports 'output_integer_quant_tflite', use it.
      2. Otherwise convert to SavedModel (float), then apply TFLiteConverter INT8.
      3. If onnx2tf outputs TFLite directly without SavedModel, warn and use float TFLite.
    """
    import tensorflow as tf
    import onnx2tf
    import glob, shutil

    calib_samples = _collect_calib_samples(test_loader, n_calib)
    output_dir = tempfile.mkdtemp()
    print(f"Converting ONNX → TFLite INT8 (calibration: {n_calib} samples)...")

    # Probe onnx2tf API to pick the right call signature
    onnx2tf_params = inspect.signature(onnx2tf.convert).parameters
    supports_int8_param = 'output_integer_quant_tflite' in onnx2tf_params

    tflite_bytes = None

    if supports_int8_param:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            output_integer_quant_tflite=True,
            representative_dataset_for_int8quant=calib_samples,
            non_verbose=True,
        )
        int8_files = glob.glob(os.path.join(output_dir, '*full_integer_quant*.tflite'))
        if int8_files:
            int8_files.sort(key=lambda x: 'full_integer_quant' not in x)
            shutil.copy(int8_files[0], tflite_path)
            with open(tflite_path, 'rb') as f:
                tflite_bytes = f.read()

    if tflite_bytes is None:
        # Either API doesn't support INT8 params, or no INT8 output was found —
        # convert to float first, then quantize via TFLiteConverter if SavedModel exists.
        if supports_int8_param:
            shutil.rmtree(output_dir)
            output_dir = tempfile.mkdtemp()
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            non_verbose=True,
        )

        saved_model_pb = os.path.join(output_dir, 'saved_model.pb')
        if os.path.exists(saved_model_pb):
            def representative_dataset():
                for s in calib_samples:
                    yield [s]
            converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_bytes = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_bytes)
        else:
            # Newer onnx2tf outputs TFLite directly — no SavedModel to re-quantize from.
            any_tflite = glob.glob(os.path.join(output_dir, '*.tflite'))
            if any_tflite:
                print("WARNING: onnx2tf produced float TFLite (no SavedModel available). "
                      "Saving as-is; upgrade onnx2tf for INT8 TFLite.")
                shutil.copy(any_tflite[0], tflite_path)
                with open(tflite_path, 'rb') as f:
                    tflite_bytes = f.read()
            else:
                raise FileNotFoundError(
                    f"onnx2tf produced no usable output in {output_dir}. "
                    f"Contents: {os.listdir(output_dir)}"
                )

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"TFLite saved: {tflite_path}  ({size_kb:.1f} KB)")
    return tflite_bytes, size_kb


def evaluate_tflite(tflite_bytes, test_loader):
    """Evaluate TFLite model (INT8 or float) accuracy on the test set."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    is_int8 = inp['dtype'] == np.int8
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    correct = total = 0
    for feats, labels in test_loader:
        for i in range(feats.shape[0]):
            x = feats[i:i+1].numpy().astype(np.float32)
            if is_int8:
                x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            interpreter.set_tensor(inp['index'], x)
            interpreter.invoke()
            logits = interpreter.get_tensor(out['index'])
            if is_int8:
                logits = (logits.astype(np.float32) - out_zp) * out_scale
            pred = np.argmax(logits)
            correct += int(pred == labels[i].item())
            total += 1
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument('--skip-tflite', action='store_true',
                        help='Skip TFLite conversion; produce INT8 ONNX only')
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

    int8_onnx_path = os.path.join(args.checkpoints_dir, f'{args.model}_int8.onnx')
    int8_onnx_kb = quantize_onnx_int8(onnx_path, int8_onnx_path, test_loader, args.n_calib)

    print("Evaluating INT8 ONNX accuracy on test set...")
    int8_onnx_acc = evaluate_int8_onnx(int8_onnx_path, test_loader)
    onnx_acc_drop = (float32_acc - int8_onnx_acc) * 100

    tflite_kb = None
    tflite_acc = None
    tflite_acc_drop = None

    if not args.skip_tflite:
        tflite_path = os.path.join(args.checkpoints_dir, f'{args.model}_int8.tflite')
        try:
            tflite_bytes, tflite_kb = convert_to_tflite_int8(
                onnx_path, tflite_path, test_loader, n_calib=args.n_calib
            )
            print("Evaluating TFLite accuracy on test set...")
            tflite_acc = evaluate_tflite(tflite_bytes, test_loader)
            tflite_acc_drop = (float32_acc - tflite_acc) * 100
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            print("Continuing with INT8 ONNX results only.")

    print(f"\n--- Results: {args.model.upper()} ---")
    print(f"  Float32 accuracy    : {float32_acc*100:.2f}%")
    print(f"  INT8 ONNX accuracy  : {int8_onnx_acc*100:.2f}%  (drop: {onnx_acc_drop:.2f}%)")
    if tflite_acc is not None:
        print(f"  TFLite accuracy     : {tflite_acc*100:.2f}%  (drop: {tflite_acc_drop:.2f}%)")
    print(f"  Float32 ONNX size   : {onnx_kb:.1f} KB")
    print(f"  INT8 ONNX size      : {int8_onnx_kb:.1f} KB")
    if tflite_kb is not None:
        print(f"  TFLite size         : {tflite_kb:.1f} KB")


if __name__ == '__main__':
    main()
