#!/bin/bash

source ~/garuda_env/bin/activate

module="mobilenet_v3"
cfgs=("cifar100" "oxford_flowers102")
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"	
)
qtypes=("fp16" "int8")

for cfg in "${cfgs[@]}"
do
	for model in "${models[@]}"
	do
		for qtype in "${qtypes[@]}"
		do
			for i in {1..5}
			do
				args=(
					--module "${module}"
					--cfg "${cfg}"
					--engine "./generated/trt/${qtype}-${cfg}-${model}-${i}.engine"
				)

				log="results/tensorrt-accs/${qtype}-${cfg}-${model}.txt"
				python3	ptq_engine_infer.py "${args[@]}" >> $log
			done
		done
	done
done


module="efficientnet"
cfgs=("cifar100" "oxford_flowers102")
models=(
	"EfficientNetB1"
	"EfficientNetB3"	
)
qtypes=("fp16" "int8")

for cfg in "${cfgs[@]}"
do
	for model in "${models[@]}"
	do
		for qtype in "${qtypes[@]}"
		do
			for i in {1..5}
			do
				args=(
					--module "${module}"
					--cfg "${cfg}"
					--engine "./generated/trt/${qtype}-${cfg}-${model}-${i}.engine"
				)

				log="results/tensorrt-accs/${qtype}-${cfg}-${model}.txt"
				python3	ptq_engine_infer.py "${args[@]}" >> $log
			done
		done
	done
done


module="densenet"
cfgs=("cifar100" "oxford_flowers102")
models=(
	"DenseNet169"
	"DenseNet201"	
)
qtypes=("fp16" "int8")

for cfg in "${cfgs[@]}"
do
	for model in "${models[@]}"
	do
		for qtype in "${qtypes[@]}"
		do
			for i in {1..5}
			do
				args=(
					--module "${module}"
					--cfg "${cfg}"
					--engine "./generated/trt/${qtype}-${cfg}-${model}-${i}.engine"
				)

				log="results/tensorrt-accs/${qtype}-${cfg}-${model}.txt"
				python3	ptq_engine_infer.py "${args[@]}" >> $log
			done
		done
	done
done
