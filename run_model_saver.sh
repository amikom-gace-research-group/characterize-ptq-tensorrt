#!/bin/bash

source ~/garuda_env/bin/activate

cfgs=("cifar100" "oxford_flowers102")
model_names=(
	"EfficientNetB1"
	"EfficientNetB3"
)

for cfg in "${cfgs[@]}"
do
	for model_name in "${model_names[@]}"
	do
		for i in {1..5}
		do
			args=(
				--model-name "${model_name}"
				--cfg "${cfg}"
				--weights "generated/${cfg}/${model_name}/${i}/checkpoint"
				--num "${i}"
			)

			python3 model_saver.py "${args[@]}"
		done
	done
done
