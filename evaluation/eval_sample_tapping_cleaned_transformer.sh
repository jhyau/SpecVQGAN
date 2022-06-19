# Path to experiment
#EXPERIMENT_PATH=./logs/2021-06-09T15-17-18_vas_transformer
#EXPERIMENT_PATH=./logs/2021-12-11T17-05-22_asmr_by_material_transformer
#EXPERIMENT_PATH=./logs/2021-11-30T07-58-56_asmr_by_material_codebook
#EXPERIMENT_PATH=./logs/2022-02-10T21-45-30_asmr_by_material_transformer
#EXPERIMENT_PATH=./logs/2022-02-19T06-10-59_asmr_by_material_transformer_no_early_stop
#EXPERIMENT_PATH=./logs/2022-02-23T07-48-24_asmr_by_material_1hr_transformer_no_early_stop
#EXPERIMENT_PATH=./logs/2022-02-24T05-38-23_ceramic_transformer_no_early_stop
EXPERIMENT_PATH=./logs/2022-06-19T05-57-36_tapping_materials_cleaned_transformer

# Select a dataset here
DATASET="VAS"
# DATASET="VGGSound"

# TOP_K_OPTIONS=( "1" "16" "64" "100" "128" "256" "512" "1024" )
TOP_K_OPTIONS=( "64" )
# VGGSOUND_SAMPLES_PER_VIDEO=10
VGGSOUND_SAMPLES_PER_VIDEO=1
# VAS_SAMPLES_PER_VIDEO=100
VAS_SAMPLES_PER_VIDEO=10

if [[ "$DATASET" == "VGGSound" ]]; then
    # EXTRACT_FILES_CMD="$EXTRACT_SPECS_VGGSOUND"
    EXTRACT_FILES_CMD="$EXTRACT_SPECS_VGGSOUND && $EXTRACT_FEATS_VGGSOUND"
    SPEC_DIR_PATH="$LOCAL_SCRATCH/melspec_10s_22050hz/"
    RGB_FEATS_DIR_PATH="$LOCAL_SCRATCH/feature_rgb_bninception_dim1024_21.5fps/"
    FLOW_FEATS_DIR_PATH="$LOCAL_SCRATCH/feature_flow_bninception_dim1024_21.5fps/"
    SAMPLES_FOLDER="VGGSound_test"
    SPLITS="\"[test, ]\""
    SAMPLER_BATCHSIZE=32
    SAMPLES_PER_VIDEO=$VGGSOUND_SAMPLES_PER_VIDEO
elif [[ "$DATASET" == "VAS" ]]; then
    # EXTRACT_FILES_CMD="$EXTRACT_SPECS_VAS"
    EXTRACT_FILES_CMD="$EXTRACT_SPECS_VAS && $EXTRACT_FEATS_VAS"
    SPEC_DIR_PATH="/juno/u/jyau/regnet/data/features/tapping/materials/melspec_10s_22050hz_melgan"
    RGB_FEATS_DIR_PATH="/juno/u/jyau/regnet/data/features/tapping/materials/feature_rgb_bninception_dim1024_21.5fps"
    FLOW_FEATS_DIR_PATH="/juno/u/jyau/regnet/data/features/tapping/materials/feature_flow_bninception_dim1024_21.5fps"
    SAMPLES_FOLDER="tapping_cleaned_transformer"
    #SPLITS="\"[validation, ]\""
    SAMPLER_BATCHSIZE=4
    SAMPLES_PER_VIDEO=$VAS_SAMPLES_PER_VIDEO
else
    echo "NotImplementedError"
    exit
fi

NOW="eval_with_melgan_tapping_cleaned"
CKPT="last"
VOCODER="./vocoder/logs/tapping_cleaned/"
# Some info to print
echo "Local Scratch" $LOCAL_SCRATCH
echo "Hostlist:" $HOSTLIST
echo "Samples per video:" $SAMPLES_PER_VIDEO "; Sampler path" $EXPERIMENT_PATH
echo "Checkpoint: " $CKPT
echo $EXTRACT_FILES_CMD
echo $SPEC_DIR_PATH
echo $RGB_FEATS_DIR_PATH
echo $FLOW_FEATS_DIR_PATH

# NOTE: nproc_per_node needs to be equal to or less than the number of GPUs available
# Originally one of the sampler commands:
# sampler.splits=$SPLITS \
# Sample
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
        evaluation/generate_samples.py \
        sampler.config_sampler=evaluation/configs/sampler.yaml \
        sampler.model_logdir=$EXPERIMENT_PATH \
        sampler.samples_per_video=$SAMPLES_PER_VIDEO \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
	sampler.ckpt=$CKPT\
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
        data.params.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
        sampler.now=$NOW \
	lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder=$VOCODER
