# Create datasets in CR1 and CR
MAX=0p60
for CAT in tthbb_0p10To${MAX} tthbb_0p20To${MAX} tthbb_0p30To${MAX} tthbb_0p40To${MAX}
do
    python scripts/dataset/parquet_to_dctr.py --cfg parameters/spanet_v2/CR1/${CAT}/data_preprocessing_CR1_${CAT}.yaml -o /eos/user/m/mmarcheg/ttHbb/dctr/training_datasets/spanet_v2/${CAT}_ttlf0p30/dataset_dctr_${CAT}_ttlf0p30.parquet
done

MAX=0p75
for CAT in tthbb_0p10To${MAX} tthbb_0p20To${MAX} tthbb_0p30To${MAX} tthbb_0p40To${MAX}
do
    python scripts/dataset/parquet_to_dctr.py --cfg parameters/spanet_v2/CR/${CAT}/data_preprocessing_CR_${CAT}.yaml -o /eos/user/m/mmarcheg/ttHbb/dctr/training_datasets/spanet_v2/${CAT}_ttlf0p30/dataset_dctr_${CAT}_ttlf0p30.parquet
done

# Create datasets in CR2
MAX=1p00
for CAT in tthbb_0p60To0p75_reweigh_tthbb_0p10To${MAX} tthbb_0p60To0p75_reweigh_tthbb_0p20To${MAX} tthbb_0p60To0p75_reweigh_tthbb_0p30To${MAX} tthbb_0p60To0p75_reweigh_tthbb_0p40To${MAX}
do
    python scripts/dataset/parquet_to_dctr.py --cfg parameters/spanet_v2/CR2/${CAT}/data_preprocessing_CR2_${CAT}.yaml -o /eos/user/m/mmarcheg/ttHbb/dctr/training_datasets/spanet_v2/${CAT}_ttlf0p30/dataset_dctr_${CAT}_ttlf0p30.parquet
done
