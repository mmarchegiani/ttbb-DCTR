LOG_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/scan_tthbb
PARAMETERS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/CR1_scan_tthbb
# Loop over two folders:
# /afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/features_26 /afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/features_8
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
        python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --log_dir $LOG_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr5e-3_cosine_annealing --good-gpus
    done
done
