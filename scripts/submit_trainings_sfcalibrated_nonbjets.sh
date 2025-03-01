LOG_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/sfcalibrated_with_ttlf_reweighting
PARAMETERS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/CR1
# Loop over two folders:
# /afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/features_26 /afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/features_8
for FEATURES_DIR in \
    features_7_nonbjets
do
    # lr = 1e-3
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr1e-3_decay1e-3.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay1e-3 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr1e-3_decay1e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay1e-4 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr1e-3_decay5e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay5e-4 --good-gpus
    # lr = 2e-3
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr2e-3_decay1e-3.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay1e-3 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr2e-3_decay1e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay1e-4 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr2e-3_decay5e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay5e-4 --good-gpus
    # lr = 5e-4
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr5e-4_decay1e-3.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay1e-3 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr5e-4_decay1e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay1e-4 --good-gpus
    python scripts/submit_to_condor.py --cfg $PARAMETERS_DIR/$FEATURES_DIR/run_parameters_lr5e-4_decay5e-4.yaml --log_dir $LOG_DIR/$FEATURES_DIR/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay5e-4 --good-gpus
done
