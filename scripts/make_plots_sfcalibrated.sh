MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/sfcalibrated_with_ttlf_reweighting
PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/CR1

for LR in 2e-3 #1e-3 5e-4
do
    #for DECAY in 1e-3 5e-4 1e-4
    for DECAY in 1e-4
    do
        for FEATURES_DIR in \
            features_26 \
            features_8
        do
        python scripts/plot/validation.py $MODELS_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr${LR}_decay${DECAY}/lightning_logs/version_0 --cfg $PARAMS_DIR/$FEATURES_DIR/run_parameters_lr${LR}_decay${DECAY}.yaml --plot_dir plots_CR1_bynjet -j 10
        done
    done
done