#!/bin/bash

source ~/garuda_env/bin/activate

module="mobilenet_v3"
cfgs=("cifar100")
model_names=(
	"MobileNetV3Small"
	"MobileNetV3Large"
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

		python3 int8_converts_hwrecords.py "${args[@]}" >> "results/int8-${cfg}-${model_name}.txt"
	done
done


module="efficientnet"
cfgs=("cifar100")
model_names=(
	"EfficientNetB1"
	"EfficientNetB3"
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

		python3 int8_converts_hwrecords.py "${args[@]}" >> "results/int8-${cfg}-${model_name}.txt"
	done
done


module="densenet"
cfgs=("cifar100")
model_names=(
	"DenseNet169"
	"DenseNet201"
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

		python3 int8_converts_hwrecords.py "${args[@]}" >> "results/int8-${cfg}-${model_name}.txt"
	done
done
