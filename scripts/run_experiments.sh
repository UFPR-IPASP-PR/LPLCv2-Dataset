run_torch() {
    echo ${1} ${2} ${3} ${4}
    for f in 0 1 2 3 4
    do
        python3 main.py \
            -c configs/models/${4}.json \
            -d 0 \
            -n ${1}_${f}_1 \
            -bs 64 \
            -dt configs/split/${2}.json \
            -f ${f}_1 \
            -p -pt test \
            -t ${3} -tt \
            -v configs/train/test_tt.json \

        python3 main.py \
            -c configs/models/${4}.json \
            -d 0 \
            -n ${1}_${f}_2 \
            -bs 64 \
            -dt configs/split/${2}.json \
            -f ${f}_2 \
            -p -pt test \
            -t ${3} -tt \
            -v configs/train/test_tt.json \

    done
}

run_torch base_old base_old configs/train/train.json resnet50
run_torch base_oup base_old_updated configs/train/train.json resnet50
run_torch base_new base_new configs/train/train.json resnet50

