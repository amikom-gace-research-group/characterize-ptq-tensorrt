#!/bin/bash

# source the environment
source ~/garuda_env/bin/activate

# vars
cfg="cifar100"
models=(
	"DenseNet201"
)

for model in "${models[@]}"
do
	logfile="results/keras-hwrecords/${cfg}-${model}-filter-10%.txt"
	args=(
		--cfg ${cfg} 
		--model-name ${model}
		--weights generated/${cfg}/${model}/1/checkpoint
		--batch 1
	)

	python3 keras_hwrecord.py "${args[@]}" >> $logfile
done
