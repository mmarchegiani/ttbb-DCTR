EPOCH=1612
python scripts/validation.py /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay5e-4/lightning_logs/version_0 --epoch $EPOCH --plot_dir plots_$EPOCH --cfg parameters/cr2/run_parameters_lr2e-3_decay5e-4.yaml
python scripts/correlation.py /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay5e-4/lightning_logs/version_0 --epoch $EPOCH --plot_dir plots_$EPOCH --cfg parameters/cr2/run_parameters_lr2e-3_decay5e-4_ttlf0p60.yaml
