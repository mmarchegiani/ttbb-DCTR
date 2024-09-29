import os
import argparse
import awkward as ak
from omegaconf import OmegaConf

from ttbb_dctr.models.binary_classifier import BinaryClassifier, WrappedModel
from ttbb_dctr.lib.data_preprocessing import get_device

def get_epoch(s):
    return int(s.split("-")[0].split("=")[-1])

def load_model(log_directory, model_class, epoch=None):
    assert "checkpoints" in os.listdir(os.path.join(log_directory)), "No checkpoints found in log directory"
    assert "hparams.yaml" in os.listdir(log_directory), "No hparams.yaml found in log directory"
    hparams = OmegaConf.load(os.path.join(log_directory, "hparams.yaml"))
    if epoch is None:
        checkpoint = os.path.join(log_directory, "checkpoints", "last.ckpt")
    else:
        checkpoints_by_epoch = [c for c in os.listdir(os.path.join(log_directory, "checkpoints")) if not "last.ckpt" in c]
        checkpoints = list(filter(lambda x : get_epoch(x) == epoch, checkpoints_by_epoch))
        assert len(checkpoints) == 1, f"Found {len(checkpoints)} checkpoints for epoch {epoch}"
        checkpoint = os.path.join(log_directory, "checkpoints", checkpoints[0])
    print("Loading model from checkpoint:", checkpoint)
    model = model_class.load_from_checkpoint(checkpoint, **hparams)
    return WrappedModel(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_directory', type=str, help="Pytorch Lightning Log directory containing the checkpoint and hparams.yaml file.")
    parser.add_argument('--output', type=str, default=None, help="Output model file path", required=False)
    parser.add_argument('--epoch', type=int, default=None, help="Select the epoch to load the model from", required=False)
    parser.add_argument('--device', type=str, default=None, choices=["cpu", "cuda"], help="Device to use for model export", required=False)
    args = parser.parse_args()

    if args.output is None:
        output_path = os.path.join(args.log_directory, "model.onnx")
    else:
        output_path = args.output
    if args.epoch is not None:
        output_path = output_path.replace(".onnx", f"_epoch{args.epoch}.onnx")

    if args.device is not None:
        device = args.device
    else:
        device = get_device()

    # Compute DCTR score and weight
    model = load_model(args.log_directory, BinaryClassifier, epoch=args.epoch)

    # Export model to ONNX
    print("Exporting model to ONNX...")
    model.eval()
    model.to(device).to_onnx(output_path,
        export_params=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={                   # Define dynamic axes for batch size
        'input': {0: 'batch_size'},  # First dimension of 'input' is dynamic
        'output': {0: 'batch_size'}  # First dimension of 'output' is dynamic
        }
    )
    print("Model exported to", output_path)

