# Closure test in CR1
MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/spanet_v2_bugfix
PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/spanet_v2/CR1
LR=5e-3

#for TTHBB_DIR in \
#    tthbb_0p10To0p60 \
#    tthbb_0p20To0p60 \
#    tthbb_0p30To0p60 \
#    tthbb_0p40To0p60
#do
#	for FEATURES_DIR in \
#            features_26 \
#            features_8
#	do
#		python scripts/plot/validation.py $MODELS_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr${LR}/lightning_logs/version_0 --cfg $PARAMS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --plot_dir plots_CR1 -j 10
#	done
#done

# Closure test in CR2 using model trained in CR1
#MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/spanet_v2_bugfix
#PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/spanet_v2/CR2
#LR=5e-3
#
#	#tthbb_0p10To1p00 \
#    #tthbb_0p20To1p00 \
#	#tthbb_0p30To1p00 \
#for TTHBB_DIR in \
#    tthbb_0p40To1p00
#do
#	for FEATURES_DIR in \
#            features_26 \
#            features_8
#	do
#		# if TTHBB_DIR is tthbb_0p40To1p00, then CR1_DIR=tthbb_0p40To0p60
#		CR1_DIR=$(echo $TTHBB_DIR | sed 's/1p00/0p60/g')
#		python scripts/plot/validation.py $MODELS_DIR/$CR1_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr${LR}/lightning_logs/version_0 --cfg $PARAMS_DIR/tthbb_0p60To0p75_reweigh_$TTHBB_DIR/$FEATURES_DIR/run_parameters_tthbb_0p60To0p75_reweigh_${TTHBB_DIR}.yaml --plot_dir plots_CR2 -j 10
#	done
#done


# Closure test in CR
MODELS_DIR=/eos/user/m/mmarcheg/ttHbb/dctr/training/spanet_v2_bugfix
PARAMS_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR/parameters/spanet_v2/CR

	#tthbb_0p10To0p75 \
    #tthbb_0p20To0p75 \
	#tthbb_0p30To0p75 \
for TTHBB_DIR in \
    tthbb_0p40To0p75
do
	for FEATURES_DIR in \
            features_26 #\
            #features_8
	do
		python scripts/plot/validation.py $MODELS_DIR/$TTHBB_DIR/$FEATURES_DIR/binary_classifier_full_Run2_batch8092_lr${LR}/lightning_logs/version_0 --cfg $PARAMS_DIR/$TTHBB_DIR/$FEATURES_DIR/run_parameters_${TTHBB_DIR}.yaml --plot_dir plots_CR_nicerange -j 10
	done
done
