#!/usr/bin/env bash
dataset=../data/AMR/amr_2.0
SUF="_ldc-tree"
DTAG=${SUF:1}
for a in $(seq 1.7 0.1 1.7); do
    echo $a
    python3 work.py --test_data ${dataset}/dev.txt.features.preproc2.json\
        --test_forests $SCR2/k-decomp/${DTAG}/dev_forests${SUF}.hdf5\
        --test_sep2frags $SCR2/k-decomp/${DTAG}/dev_sep2frags${SUF}.pkl\
        --test_batch_size 44444\
        --load_path $1\
        --beam_size 12\
        --alpha $a\
        --max_time_step 100\
        --output_suffix _dev_out
    #python3 postprocess.py --golden_file ../data/AMR/amr_2.0/dev.txt.features\
    #    --pred_file ${1}_dev_out \
    #    --output
done
