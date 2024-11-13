MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/sfcalibrated
PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated

for LR in 2e-3 1e-3 5e-4
do
    for DECAY in 1e-3 5e-4 1e-4
    do
        #cp $PARAMS_DIR/run_parameters_lr${LR}_decay${DECAY}.yaml $PARAMS_DIR/run_parameters_lr${LR}_decay${DECAY}_ttlf0p30.yaml
        python scripts/validation.py $MODELS_DIR/binary_classifier_26features_full_Run2_batch8092_lr${LR}_decay${DECAY}/lightning_logs/version_0 --cfg $PARAMS_DIR/features_26/run_parameters_lr${LR}_decay${DECAY}.yaml
        python scripts/validation.py $MODELS_DIR/binary_classifier_26features_full_Run2_batch8092_lr${LR}_decay${DECAY}/lightning_logs/version_1 --cfg $PARAMS_DIR/features_8/run_parameters_lr${LR}_decay${DECAY}.yaml
        #python scripts/correlation.py $MODELS_DIR/binary_classifier_26features_full_Run2_batch8092_lr${LR}_decay${DECAY}/lightning_logs/version_0 --cfg $PARAMS_DIR/run_parameters_lr${LR}_decay${DECAY}_ttlf0p30.yaml
    done
done