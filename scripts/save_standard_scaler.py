import os
import argparse
import awkward as ak

from omegaconf import OmegaConf

from ttbb_dctr.lib.data_preprocessing import get_input_features, fit_standard_scaler, save_standard_scaler

def save_config(cfg, filename):
    with open(filename, "w") as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-l', '--log_dir', type=str, help="Output folder", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_model = cfg["model"]
    cfg_training = cfg["training"]

    events_train = ak.from_parquet(cfg_training["training_file"])
    events_test = ak.from_parquet(cfg_training["test_file"])
    input_features_train = get_input_features(events_train, only=cfg_training.get("input_features", None))
    input_features_test = get_input_features(events_test, only=cfg_training.get("input_features", None))
    # Fit standard scaler on training data and apply it to both training and test data. Save the standard scaler to the log directory in the subfolder `standard_scaler`
    standard_scaler = fit_standard_scaler(input_features_train)
    nevents_train = len(events_train)
    filename = os.path.join(args.log_dir, "standard_scaler", f"standard_scaler_train_{nevents_train}.pkl")
    save_standard_scaler(standard_scaler, filename)
