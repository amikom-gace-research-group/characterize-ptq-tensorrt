#!/bin/bash

source ~/garuda_env/bin/activate

module="mobilenet_v3"
cfgs=("cifar100")
model_names=(
	"MobileNetV3Small"
)

for cfg in "${cfgs[@]}"
do
	for model_name in "${model_names[@]}"
	do
		args=(
			--cfg "${cfg}"
			--model-name "${model_name}"
			--module "${module}"
		)

		python3 mod_int8_converts_hwrecords.py "${args[@]}"
	done
done
