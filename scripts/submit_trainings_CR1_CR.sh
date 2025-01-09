LOG_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/spanet_v2
PARAMETERS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/spanet_v2/CR1

for TTHBB_DIR in \
    tthbb_0p10To0p60 \
    tthbb_0p20To0p60 \
    tthbb_0p30To0p60 \
    tthbb_0p40To0p60
do
    for FEATURES_DIR in \
        features_26 \
        features_8
    do
        python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --log_dir $LOG_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr5e-3 --good-gpus --ngpu 4 --ncpu 12
    done
done

PARAMETERS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/spanet_v2/CR

for TTHBB_DIR in \
    tthbb_0p10To0p75 \
    tthbb_0p20To0p75 \
    tthbb_0p30To0p75 \
    tthbb_0p40To0p75
do
    for FEATURES_DIR in \
        features_26 \
        features_8
    do
        python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --log_dir $LOG_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr5e-3 --good-gpus --ngpu 4 --ncpu 12
    done

done