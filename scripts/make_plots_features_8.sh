for EPOCH in {450..700..50}
do
    python scripts/validation.py /eos/user/m/mmarcheg/ttHbb/dctr/training/features_8/binary_classifier_8features_full_Run2_batch8092_lr1e-3/lightning_logs/version_2 --epoch $EPOCH --plot_dir plots_$EPOCH --cfg parameters/features_8/run_parameters_lr1e-3.yaml
    python scripts/correlation.py /eos/user/m/mmarcheg/ttHbb/dctr/training/features_8/binary_classifier_8features_full_Run2_batch8092_lr1e-3/lightning_logs/version_2 --epoch $EPOCH --plot_dir plots_$EPOCH --cfg parameters/features_8/run_parameters_lr1e-3_ttlf0p60.yaml
done
