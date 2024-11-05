# lr = 1e-3
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr1e-3_decay1e-3.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay1e-3 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr1e-3_decay1e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay1e-4 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr1e-3_decay5e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr1e-3_decay5e-4 --good-gpus
# lr = 2e-3
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr2e-3_decay1e-3.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay1e-3 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr2e-3_decay1e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay1e-4 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr2e-3_decay5e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr2e-3_decay5e-4 --good-gpus
# lr = 5e-4
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr5e-4_decay1e-3.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay1e-3 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr5e-4_decay1e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay1e-4 --good-gpus
python scripts/submit_to_condor.py --cfg parameters/cr2/run_parameters_lr5e-4_decay5e-4.yaml --log_dir /eos/user/m/mmarcheg/ttHbb/dctr/training/cr2/binary_classifier_26features_full_Run2_batch8092_lr5e-4_decay5e-4 --good-gpus
