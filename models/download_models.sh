#!/bin/bash

# Set models and datasets to download
models=(
    "sentence-transformers/all-MiniLM-L6-v2"
    "THUDM/chatglm-6b"
)
#datasets=("Matthijs/cmu-arctic-xvectors")

# Set the current directory
CURRENT_DIR=$(pwd)

# Download models
for model in "${models[@]}"; do
	echo "----- Downloading from https://huggingface.co/${model} -----"
	if [ -d "${model}" ]; then
		(cd "${model}" && git pull && git lfs pull)
	else
		git clone --recurse-submodules "https://huggingface.co/${model}" "${model}"
	fi
done

# Download datasets
#for dataset in "${datasets[@]}"; do
#	echo "----- Downloading from https://huggingface.co/datasets/${dataset} -----"
#	if [ -d "${dataset}" ]; then
#		(cd "${dataset}" && git pull && git lfs pull)
#	else
#		git clone --recurse-submodules "https://huggingface.co/datasets/${dataset}" "${dataset}"
#	fi
#done