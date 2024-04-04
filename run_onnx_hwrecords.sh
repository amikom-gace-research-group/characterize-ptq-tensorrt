#!/bin/bash

source ~/garuda_env/bin/activate

cfgs=("oxford_flowers102" "cifar100")
model_names=(
	"MobileNetV3Small"
	"MobileNetV3Large"
	"EfficientNetB1"
	"EfficientNetB3"
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
		)

		python3 onnx_hwrecords.py "${args[@]}" >> "results/onnx/${cfg}-${model_name}.txt"
	done
done
