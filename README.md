# ttbb-DCTR
Differential correction of the irreducible tt+bb background in the ttH(bb) semileptonic analysis with DNNs using classification for tuning and reweighting (DCTR) [[1907.08209](https://arxiv.org/abs/1907.08209)]

## Installation
Create `micromamba` environment and install the `ttbb_dctr` package.
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) # Install micromamba in your system
micromamba create -n ttbb-dctr python=3.11 -c conda-forge
micromamba activate ttbb-dctr

git clone git@github.com:mmarchegiani/ttbb-DCTR.git
cd ttbb-DCTR
pip install -e .
```

In addition, the `ttbb_dctr` package depends on the external `tthbb_spanet` package. Install it in the same environment:
```
cd ..
git clone git@github.com:mmarchegiani/ttHbb_SPANet.git
cd ttHbb_SPANet
pip install -e .
```
## Data preprocessing

### Configuration file
See [data_preprocessing_CR1_scan_tthbb_compute_njet_reweighting.yaml](https://github.com/mmarchegiani/ttbb-DCTR/blob/main/parameters/spanet_v2/CR1/data_preprocessing_CR1_scan_tthbb_compute_njet_reweighting.yaml) for an example configuration file.

### Parameters for SPANet inference
It is possible to specify a model of SPANet in order to do the inference and extract the classification scores. Set the following parameters in the YAML:
```
preprocessing:
  spanet:
      model: /path/to/spanet_model.onnx
      event_file: /path/to/event_file.yaml
```

### Compute njet reweighting
Before running the dataset production, it is necessary to compute the 1D reweighting based on the njet distribution. This is done with the following command:
```
python scripts/dataset/parquet_to_dctr.py --cfg data_preprocessing.yaml -o dataset_dctr.parquet --compute-njet-reweighting
```
This will save a file `dataset_dctr_reweighting_map.yaml` containing the 1D reweighting map. Store this file in a dedicated folder because it is the 1D reweighting map used to reweigh events during the training dataset production.

### Create training dataset
Here is an example of how to run the training dataset creation:
```
python scripts/dataset/parquet_to_dctr.py --cfg parameters/spanet_v2/CR1/${CAT}/data_preprocessing_CR1_${CAT}.yaml -o /eos/user/m/mmarcheg/ttHbb/dctr/training_datasets/spanet_v2/${CAT}_ttlf0p30/dataset_dctr_${CAT}_ttlf0p30.parquet
```
where the `--cfg` corresponds to a config file in YAML format where all the necessary parameters are specified. `-o` indicates the path of the output parquet file.
See also an example [wrapper script](https://github.com/mmarchegiani/ttbb-DCTR/blob/main/scripts/create_dctr_datasets_spanet_v2.sh) to run the dataset production in multiple categories.

## Training
Example training script:
```
python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --log_dir $LOG_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr5e-3 --good-gpus --ngpu 4 --ncpu 12
```
See also an example [wrapper script](https://github.com/mmarchegiani/ttbb-DCTR/blob/main/scripts/submit_trainings_CR1_CR.sh) to submit multiple trainings with Condor jobs.

## Save standard scaler
Since in the training script there is standard scaling applied at the inputs, the same scaling needs to be applied when evaluating the model, therefore we save the standard scaler to a pickle file.

> [!TIP]
> Possible improvement: save the parquet file already with standard scaling applied at the inputs and remove this extra step.

## Plots
TBC.
