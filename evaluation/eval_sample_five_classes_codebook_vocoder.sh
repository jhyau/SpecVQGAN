# Path to experiment
#EXPERIMENT_PATH=./logs/2021-06-09T15-17-18_vas_transformer
#EXPERIMENT_PATH=./logs/2021-12-11T17-05-22_asmr_by_material_transformer
#EXPERIMENT_PATH=./logs/2021-11-30T07-58-56_asmr_by_material_codebook
#EXPERIMENT_PATH=./logs/2022-02-04T21-23-39_asmr_by_material_juno_codebook/
#EXPERIMENT_PATH=./logs/2022-02-02T03-12-20_reproduce_vas_codebook
EXPERIMENT_PATH=./logs/2022-06-11T15-48-36_five_class_tapping_materials_codebook

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
    SAMPLES_FOLDER="five_class_tapping_codebook"
    #SPLITS="\"[validation, ]\""
    SAMPLER_BATCHSIZE=4
    SAMPLES_PER_VIDEO=$VAS_SAMPLES_PER_VIDEO
else
    echo "NotImplementedError"
    exit
fi

NOW="eval_with_melgan_five_class_epochs_70000"
CKPT="last"
VOCODER="./vocoder/logs/five_classes_tapping_epochs_70000/"
# Some info to print
echo "Local Scratch" $LOCAL_SCRATCH
echo "Hostlist:" $HOSTLIST
echo "Samples per video:" $SAMPLES_PER_VIDEO "; Sampler path" $EXPERIMENT_PATH
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
	sampler.ckpt=$CKPT \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        sampler.now=$NOW \
	lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder=$VOCODER