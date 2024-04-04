#!/bin/bash

source ~/garuda_env/bin/activate

cfgs=("cifar100")
model_names=(
	"MobileNetV3Small"
	"EfficientNetB3"
	"DenseNet201"
)

for cfg in "${cfgs[@]}"
do
	for model_name in "${model_names[@]}"
	do
		args=(
			--cfg "${cfg}"
			--model-name "${model_name}"
		)

		python3 mod_fp16_converts_hwrecords.py "${args[@]}"
	done
done
