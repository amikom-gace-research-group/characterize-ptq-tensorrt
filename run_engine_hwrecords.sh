#!/bin/bash

source ~/garuda_env/bin/activate

models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
	"EfficientNetB1"
	"EfficientNetB3"
	"DenseNet169"
	"DenseNet201"
)
datasets=("cifar100" "oxford_flowers102")
qtypes=("fp16" "int8")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
		for qtype in "${qtypes[@]}"
		do
			args=(
				--engine "generated/trt/${qtype}-${dataset}-${model}-1.engine"
			)

			log="results/engine-hwrecords/${qtype}-${dataset}-${model}.txt"
			python3 engine_hwrecords.py "${args[@]}" >> $log
		done
	done
done

