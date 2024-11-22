MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/scan_tthbb
PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/dctr_sfcalibrated/CR1_scan_tthbb
LR=2e-3

for TTHBB_CUT in tthbb_0p10To0p60 tthbb_0p20To0p60 tthbb_0p30To0p60 tthbb_0p40To0p60
do
	for FEATURES_DIR in \
            features_26 \
            features_8
	do
		python scripts/plot/validation.py $MODELS_DIR/$TTHBB_CUT/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr${LR}/lightning_logs/version_0 --cfg $PARAMS_DIR/$TTHBB_CUT/$FEATURES_DIR/run_parameters_${TTHBB_CUT}.yaml --plot_dir plots_CR1 -j 10
	done
done
