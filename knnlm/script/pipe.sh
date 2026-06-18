set -e
cd $(dirname $0)

options=$(getopt -o m -l action:,dataset:,ckpt-name:,date:,prefix:,size:,gpu:,pt:,pr:,stop:,dstore-fp16,full,analyze,data-type:,count-acc,qa:,deviation,continual:,continual-prefix:,amlt,use-k:,end-task:,end-task-data-path:,k-shot:,net,test-name:,num-shards:,shard-id:,lmbda:,knn-temp: -- "$@")
[ $? -eq 0 ] || { 
    echo "Incorrect options provided"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
        --action)
            shift; 
            action=$1
            # [[ ! ${action} =~ save|index|infer|nothing ]] && {
            #     echo "Incorrect actions provided"
            #     exit 1
            # }
            ;;
        --ckpt-name)
            shift; 
            ckpt_name=$1
            ;;
        --dataset)
            shift; 
            dataset=$1
            ;;
        --date)
            shift; 
            date=$1
            ;;
        --prefix)
            shift; 
            prefix=$1
            ;;
        --size)
            shift; 
            size=$1
            ;;
        --gpu)
            shift; 
            gpu=$1
            ;;
        --stop)
            shift; 
            stop=$1
            ;;
        --pt)
            shift;
            prune_threshold=$1
            ;;
        --pr)
            shift; 
            prune_rate=$1
            ;;
        --dstore-fp16)
            dstore_fp16=--dstore-fp16
            fp=16
            ;;
        --full)
            full=/full
            ;;
        --analyze)
            analyze_knn=--analyze-knn
            ;;
        --data-type)
            shift;
            data_type=$1
            if [[ ( -z ${data_type} ) || ( ${data_type} == 'None' ) ]]; then
                data_type=
            else
                data_type=${data_type}.
            fi
            ;;
        --count-acc)
            count_acc=--count-acc
            ;;
        --qa)
            shift;
            qa=$1
            ;;
        --deviation)
            deviation=--deviation
            ;;
        --continual)
            shift;
            continual=$1
            ;;
        --continual-prefix)
            shift;
            continual_prefix=$1
            ;;
        --amlt)
            amlt="True"
            ;;
        --use-k)
            shift;
            use_k=$1
            ;;
        --end-task)
            shift;
            end_task=$1
            ;;
        --end-task-data-path)
            shift;
            end_task_data_path=$1
            ;;
        --k-shot)
            shift;
            k_shot=$1
            ;;
        --net)
            net="True"
            ;;
        --test-name)
            shift;
            test_name=$1
            ;;
        --num-shards)
            shift;
            num_shards=$1
            ;;
        --shard-id)
            shift;
            shard_id=$1
            ;;
        --lmbda)
            shift;
            lmbda=$1
            ;;
        --knn-temp)
            shift;
            temp=$1
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "$1 is not an option"
            exit 1
            ;;
    esac
    shift
done

if [[ -z ${fp} ]]; then
    fp=32
fi

echo "Date:" ${date}
if [[ ${continual} == "False" ]]; then
    last_date=
    echo "First day"
elif [[ ${date} =~ ^[0-9]+$ ]]; then
    last_date=$(( ${date} - 1 ))
fi
if [[ (${dataset} == 'domain') && (${date} == 1) ]]; then
    last_date=180
fi
echo "Last day: ${last_date}"

fairseq_path=..
monim_path=../../monim
if [[ ${amlt} == "True" ]]; then
    echo "I'm on the amlt!"
    echo $(pwd)
    data_dir=/mnt/dstore/data/knnlm
    storage_path=/mnt/storing/data/knnlm
else
    data_dir=../../data/knnlm
    storage_path=../../data/knnlm
fi

data_path=${data_dir}/datasets/${dataset}/${data_type}${date}
text_bin=${data_path}/bin
total_prefix=${ckpt_name}_${data_type}${date}_${prefix}_fp${fp}_p_${prune_threshold},${prune_rate}

ckpt=${data_dir}/checkpoints/${ckpt_name}/checkpoint_best.pt
dstore_path=${storage_path}/storage/${ckpt_name}/${dataset}
dstore=${dstore_path}/dstore/${total_prefix}
index=${dstore_path}/knn/${total_prefix}.index
output_dir=${data_dir}/output

mkdir -p ${dstore_path}/dstore
mkdir -p ${dstore_path}/knn

net_data_path=${data_path}/net/total
net_feature_path=${net_data_path}/${ckpt_name}/${prune_threshold}_${prune_rate}
cache_path=${storage_path}/datasets/${dataset}/${data_type}${date}/net/total/${ckpt_name}/${prune_threshold}_${prune_rate}
mkdir -p ${cache_path}

seed=927
hid=128
nl=4
bs=64
drop=0.2
arch='metak'
feature="all"
feature_str=$(printf "$feature" | tr , _)
ngram=0
lr=0.0001
l1=0.05
net_path=${net_feature_path}/checkpoint/${arch}.l1.${l1}.ngram${ngram}.hid${hid}.nl${nl}.bs${bs}.drop${drop}.ft${feature_str}.seed${seed}.use_k${use_k}.metadim1
net_ckpt=${net_path}/checkpoint_best.pt

if [[ -n ${last_date} ]]; then
    old_prefix="${ckpt_name}_${data_type}${last_date}_${prefix}_fp${fp}_p_${prune_threshold},${prune_rate}"
    if [[ (${dataset} == 'domain') && (${date} == '1') ]]; then
        echo "Domain"
        blob_dstore_path=${storage_path}/storage/${ckpt_name}/daily
        last_data_path=${data_dir}/datasets/daily/${data_type}${last_date}
    else
        blob_dstore_path=${storage_path}/storage/${ckpt_name}/${dataset}
        last_data_path=${data_dir}/datasets/${dataset}/${data_type}${last_date}
    fi
    old_dstore_indisk=${blob_dstore_path}/dstore/${old_prefix}
    old_index=${blob_dstore_path}/knn/${old_prefix}.index

    old_net_ckpt=$last_data_path/net/total/${ckpt_name}/${prune_threshold}_${prune_rate}/checkpoint/${arch}.l1.${l1}.ngram${ngram}.hid${hid}.nl${nl}.bs${bs}.drop${drop}.ft${feature_str}.seed${seed}.use_k${use_k}.metadim1/checkpoint_best.pt
fi

max_tokens=1024
context_window=512
dim=
if [[ ${ckpt_name} == "gpt2-small" ]]; then
    dim=768
elif [[ ${ckpt_name} == "gpt2-medium" ]]; then
    dim=1024
elif [[ ${ckpt_name} == "gpt2-large" ]]; then
    dim=1280
elif [[ ${ckpt_name} == "wt103" ]]; then
    dim=1024
    max_tokens=3072
    context_window=1536
elif [[ ${ckpt_name} == "cc-news" ]]; then
    dim=1280
fi

if [[ -n ${count_acc} ]]; then
    softmax_batch=128
else
    softmax_batch=1024
fi

if [[ -n ${qa} ]]; then
    if [[ ${qa} == "CommonQA" ]]; then
        text_bin=${data_dir}/datasets/CommonQA/1.shot${k_shot}/bin  
    else
        text_bin=${data_dir}/datasets/realtime/${qa}.shot${k_shot}/bin
    fi
    mode=eos
    context_window=0
else
    mode=none
fi

if [[ -n ${net} ]]; then
    if [[ ${action} == "save" ]]; then
        net_command="--ar-ckpt ${old_net_ckpt}
                --ar-freq-dict ${last_data_path}
                --ar-feat-type ctxt,freq,lm_ent,lm_max,fert"
    else
        net_command="--ar-ckpt ${net_ckpt}
                --ar-freq-dict ${data_path}
                --ar-feat-type ctxt,freq,lm_ent,lm_max,fert"
    fi
fi

debug="--ckpt-name ${ckpt_name} --date ${date}"

if [[ ${action} == "save" ]];
then
    if [[ -n ${last_date} ]]; then
        continual_save_command="--knnlm --infer-dstore-path ${old_dstore_indisk} --index-file ${old_index} --no-load-keys --knn-sim-func do_not_recomp_l2 --k 1024 --probe 32 --gpu-index True --continual  --continual-prefix ${continual_prefix:-None}"
    else
        continual_save_command=
    fi
    if [[ ${continual_prefix} == "improve" ]]; then
        continual_save_command="${continual_save_command} --analyze-knn"
        echo ${continual_save_command}
    fi
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${text_bin} \
        --path ${ckpt} \
        --sample-break-mode none --max-tokens ${max_tokens} \
        --softmax-batch ${softmax_batch} --gen-subset train \
        --context-window ${context_window} \
        --save-dstore-path ${dstore} --knn-keytype 'last_ffn_input' \
        --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
        --save-knnlm-dstore --fp16 \
        --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
        --stop-size "${stop}" \
        --output-dir ${output_dir} \
        --num-shards ${num_shards} --shard-id ${shard_id} \
        ${dstore_fp16} ${analyze_knn} ${continual_save_command} ${net_command} ${debug}
    if [[ -e ${index} ]]; then
        echo "There are expired indexes. Remove them"
        rm -rf ${index}*
    fi
elif [[ ${action} == "merge" ]];
then
    python merge_dstore.py --old-dstore "${old_dstore_indisk}" --new-dstore ${dstore} --dimension ${dim} ${dstore_fp16} --output-dir ${output_dir} --num-shards ${num_shards}
elif [[ ${action} == "build_index" ]];
then
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/build_index.py \
    --dstore-path ${dstore} \
    --index-file ${index} \
    --dimension ${dim} \
    --output-dir ${output_dir} \
    ${dstore_fp16}
elif [[ ${action} == "add_index" ]];
then
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/add_index.py \
    --dstore-path ${dstore} \
    --index-file ${index} \
    --dimension ${dim} \
    --output-dir ${output_dir} \
    --num-shards ${num_shards} --shard-id ${shard_id} \
    ${dstore_fp16}
elif [[ ${action} == "merge_index" ]];
then
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/merge_index.py \
    --dstore-path ${dstore} \
    --index-file ${index} \
    --dimension ${dim} \
    --output-dir ${output_dir} \
    --num-shards ${num_shards} \
    ${dstore_fp16}
elif [[ ${action} == 'build_features' ]]; then
    if [[ ! -e ${data_path}/fertility_cache_id.pickle ]]; then
        echo "compute frequency/fertility dictionary"
        python ${monim_path}/adaptive_retrieval/cache_freq_fertility.py \
            --data ${data_path}/train.bpe \
            --load-cache "${last_data_path}" \
            --cache ${data_path} \
            --dict-path ${data_path}/bin/dict.txt \
            --csize 1
    fi
elif [[ ${action} == 'save_features' ]]; then
    split="train test"
    for s in ${split}; do
        echo "save all features on the new retrieval adaptor ${s} data"
        CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${net_data_path}/bin \
            --path ${ckpt} \
            --sample-break-mode ${mode} --max-tokens ${max_tokens} \
            --context-window ${context_window} --softmax-batch 1024 \
            --gen-subset ${s} --infer-dstore-path ${dstore} \
            --index-file ${index} \
            --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
            --k 1024 --knn-keytype last_ffn_input \
            --probe 32 --fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
            --save-feature ${cache_path}/${s} --ar-feat-cache ${data_path} \
            --knnlm \
            --dstore-fp16 \
            --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
            --gpu-index "True" \
            --num-shards ${num_shards} --shard-id ${shard_id} \
            --output-dir ${output_dir} \
            --analyze-knn
    done
elif [[ ${action} == 'train_net' ]]; then
    train="$cache_path/train"
    val="$cache_path/test"    
    mkdir -p ${net_path}

    while ! CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.run --nproc_per_node=${num_shards} ${monim_path}/adaptive_retrieval/my_ar.py \
        --hidden-units ${hid} \
        --nlayers ${nl} \
        --dropout ${drop} \
        --seed ${seed} \
        --output-dir ${net_path} \
        --lr ${lr} \
        --feature-type ${feature} \
        --batch-size ${bs} \
        --arch ${arch} \
        --l1 ${l1} \
        --train ${train} \
        --val ${val} \
        --ngram ${ngram} \
        --use-k ${use_k} \
        --cache-path ${cache_path}; do
            lr=$( python -c "print(${lr}/2)" )
            echo "Lr too big. Reset lr to ${lr}. Retrying..."
            sleep 5
    done
    # CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${data_path}/bin \
    #     --path ${ckpt} \
    #     --sample-break-mode none --max-tokens ${max_tokens} \
    #     --context-window ${context_window} --softmax-batch 1024 \
    #     --gen-subset test --infer-dstore-path ${dstore} \
    #     --index-file ${index} \
    #     --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    #     --k 1024 --knn-keytype last_ffn_input \
    #     --probe 32 --knnlm --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    #     --move-dstore-to-mem \
    #     --gpu-index "True" \
    #     --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
    #     --ar-ckpt ${net_ckpt} \
    #     --ar-freq-dict ${data_path} \
    #     --ar-feat-type ctxt,freq,lm_ent,lm_max,fert --analyze-knn

    rm -rf ${cache_path}/json
    rm -rf ${cache_path}/*.jsonl
    rm -rf ${dstore}_*.today*
    rm -rf ${index}.[0-9]
    # rm -rf ${dstore}_*vals*
elif [[ ${action} == 'infer_one' ]];
then
    dstore_path=${data_dir}/storage/${ckpt_name}/${dataset}
    dstore=${dstore_path}/dstore/${total_prefix}
    if [[ ! -e ${index} ]]; then
        index=${dstore_path}/knn/${total_prefix}.index
    fi
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${text_bin} \
        --path ${ckpt} \
        --sample-break-mode ${mode} --max-tokens ${max_tokens} \
        --softmax-batch ${softmax_batch} --gen-subset test \
        --context-window "${context_window}" \
        --infer-dstore-path ${dstore} --index-file ${index} --knn-keytype 'last_ffn_input' \
        --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
        --knnlm --fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" --k 1024 --probe 32 \
        --gpu-index "True" \
        --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
        --stop-size "${stop}" \
        --lmbda ${lmbda} --knn-temp ${temp} \
        --output-dir ${output_dir} \
        ${dstore_fp16} ${analyze_knn} ${count_acc} --qa "${qa}" ${deviation} ${net_command} ${debug}
elif [[ ${action} == 'infer' ]];
then
    # rm -rf ${output_dir}/debug/ppl/${total_prefix}.txt
    # rm -rf ${output_dir}/debug/pwd/${total_prefix}.txt

    dstore_path=${data_dir}/storage/${ckpt_name}/${dataset}
    dstore=${dstore_path}/dstore/${total_prefix}
    if [[ ! -e ${index} ]]; then
        index=${dstore_path}/knn/${total_prefix}.index
    fi
    for lmbda in $(seq 0.4 0.05 0.6); do
        for temp in $(seq 8.5 0.5 15.5); do
            CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${text_bin} \
                --path ${ckpt} \
                --sample-break-mode ${mode} --max-tokens ${max_tokens} \
                --softmax-batch ${softmax_batch} --gen-subset test \
                --context-window "${context_window}" \
                --infer-dstore-path ${dstore} --index-file ${index} --knn-keytype 'last_ffn_input' \
                --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
                --knnlm --fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" --k 1024 --probe 32 \
                --gpu-index "True" \
                --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
                --stop-size "${stop}" \
                --lmbda ${lmbda} --knn-temp ${temp} \
                --output-dir ${output_dir} \
                ${dstore_fp16} ${analyze_knn} ${count_acc} --qa "${qa}" ${deviation} ${net_command} ${debug}
        done
    done
elif [[ ${action} == 'orig_infer' ]];
then
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${text_bin} \
        --path ${ckpt} \
        --sample-break-mode ${mode} --max-tokens ${max_tokens} \
        --softmax-batch ${softmax_batch} --gen-subset test \
        --context-window ${context_window} \
        --output-dir ${output_dir} \
        --infer-dstore-path ${dstore} \
        --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
        ${analyze_knn} ${count_acc} --qa "${qa}" ${deviation}
elif [[ ${action} == 'end_task' ]];
then
    lmbda=0.25
    temp=0.5
    mode=eos
    dstore_path=${data_dir}/storage/${ckpt_name}/${dataset}
    dstore=${dstore_path}/dstore/${total_prefix}
    if [[ ! -e ${index} ]]; then
        index=${dstore_path}/knn/${total_prefix}.index
    fi
    CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${end_task_data_path}/${k_shot}/bin \
        --path ${ckpt} \
        --sample-break-mode ${mode} --max-tokens ${max_tokens} \
        --softmax-batch ${softmax_batch} --gen-subset test \
        --context-window ${context_window} \
        --infer-dstore-path ${dstore} --index-file ${index} --knn-keytype 'last_ffn_input' \
        --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
        --knnlm --fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" --k 1024 --probe 32 \
        --lmbda ${lmbda} --knn-temp ${temp} \
        --gpu-index "True" \
        --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
        --stop-size "${stop}" \
        --output-dir ${output_dir} \
        ${dstore_fp16} ${count_acc} \
        --end-task ${end_task} --end-task-data-path ${end_task_data_path} --k-shot ${k_shot} --count-acc ${net_command}
elif [[ ${action} == 'test' ]];
then
    dstore_path=${data_dir}/storage/${ckpt_name}/${dataset}
    dstore=${dstore_path}/dstore/${total_prefix}
    if [[ ! -e ${index} ]]; then
        index=${dstore_path}/knn/${total_prefix}.index
    fi
    if [[ ${test_name} == "month" ]]; then
        begin=1
        end=6
        test_name=${test_name}.
    elif [[ ${test_name} == "wiki_event" ]]; then
        begin=1
        end=3
        dataset=${test_name}
        test_name=
    elif [[ ${test_name} == "wikitext-gpt2" ]]; then
        begin=1
        end=1
        dataset=${test_name}
        test_name=
    fi
    for test_date in $(seq ${begin} 1 ${end}); do
        text_bin=${data_dir}/datasets/${dataset}/${test_name}${test_date}/bin
        CUDA_VISIBLE_DEVICES=${gpu} python ${fairseq_path}/eval_lm.py ${text_bin} \
            --path ${ckpt} \
            --sample-break-mode ${mode} --max-tokens ${max_tokens} \
            --softmax-batch ${softmax_batch} --gen-subset test \
            --context-window "${context_window}" \
            --infer-dstore-path ${dstore} --index-file ${index} --knn-keytype 'last_ffn_input' \
            --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
            --knnlm --fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" --k 1024 --probe 32 \
            --gpu-index "True" \
            --prune-threshold ${prune_threshold} --prune-rate ${prune_rate} \
            --stop-size "${stop}" \
            --output-dir ${output_dir} \
            ${dstore_fp16} ${analyze_knn} ${count_acc} --qa "${qa}" ${deviation} ${net_command} ${debug}
    done
elif [[ ${action} == 'checkpoint' ]]; then
    echo "Copying..."
    cp ${dstore}_keys.npy ${dstore}_keys.npy.copy
    echo "Copy finished"
elif [[ ${action} == 'nothing' ]]; then
    echo 'do nothing'
fi
