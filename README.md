# Training Parler-TTS for emotion ability 
# Highlight: These pipeline was embedded with the monitor function to trace the training process (particularly for emotion ability)

However, this project is still in progress, where some bugs has not been fixed.

```
#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc068-pool4
#SBATCH --job-name=first_running_with_e2v_emo_finetune_eval_with_same_sent_tianqi30
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Load the required modules
source /work/tc068/tc068/tianqi30/myvenvs/parler/bin/activate

export HF_HOME=/work/tc068/tc068/tianqi30/.cache/huggingface
export TORCH_HOME=/work/tc068/tc068/tianqi30/.cache/torch
export MODELSCOPE_CACHE=/work/tc068/tc068/tianqi30/.cache/modelscope

# Offline mode settings
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE_MODE=1

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_CACHE_DIR=/work/tc068/tc068/tianqi30/.cache/wandb
export WANDB_DATA_DIR=/work/tc068/tc068/tianqi30/.cache/wandb/data
export WANDB_PROJECT=parler-tts-emotion
export WANDB_RUN_NAME=emotion-finetune-$(date +%Y%m%d_%H%M%S)

cd /work/tc068/tc068/tianqi30/parler-tts/

JOB_TAG="speed_test_formal"
STAMP="$(date +%Y%m%d_%H%M)"
RUN_ID="${JOB_TAG}_${STAMP}"
ROOT="task_and_stuff/${RUN_ID}"

# Create output directories
mkdir -p "${ROOT}/generated_audios_emotion"
mkdir -p "${ROOT}/output_emotion_training"
mkdir -p "${ROOT}/audio_code_tmp"
mkdir -p "${ROOT}/tmp_dataset_audio"

accelerate launch ./training_parler_emotion/run_parler_tts_training.py \
    --model_name_or_path "parler-tts/parler_tts_mini_v0.1" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "parler-tts/parler_tts_mini_v0.1" \
    --prompt_tokenizer_name "parler-tts/parler_tts_mini_v0.1" \
    --report_to "wandb" \
    --overwrite_output_dir true \
    --train_dataset_name "ylacombe/expresso+reach-vb/jenny_tts_dataset+blabble-io/libritts_r+blabble-io/libritts_r" \
    --train_metadata_dataset_name "reach-vb/expresso-tagged-w-speech-mistral-v3+ylacombe/jenny-tts-10k-tagged+parler-tts/libritts_r_tags_tagged_10k_generated+parler-tts/libritts_r_tags_tagged_10k_generated" \
    --train_dataset_config_name "read+default+clean+other" \
    --train_split_name "train+train[:20%]+test.clean+test.other" \
    --eval_dataset_name "./task_and_stuff/ParlerEmotionTest2" \
    --eval_split_name "eval" \
    --eval_dataset_config_name "default" \
    --eval_metadata_dataset_name "None" \
    --emotion_column_name "style" \
    --target_audio_column_name "audio" \
    --description_column_name "text_description" \
    --prompt_column_name "text" \
    --max_eval_samples 70 \
    --per_device_eval_batch_size 4 \
    --max_duration_in_seconds 30.0 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 400 \
    --preprocessing_num_workers 2 \
    --do_train true \
    --num_train_epochs 12 \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --learning_rate 0.00008 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --logging_steps 2 \
    --freeze_text_encoder true \
    --audio_encoder_per_device_batch_size 8 \
    --dtype "bfloat16" \
    --seed 456 \
    --output_dir "./${ROOT}/output_emotion_training" \
    --temporary_save_to_disk "./${ROOT}/audio_code_tmp" \
    --save_to_disk "./${ROOT}/tmp_dataset_audio" \
    --dataloader_num_workers 4 \
    --do_eval true \
    --predict_with_generate true \
    --include_inputs_for_metrics true \
    --group_by_length true \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --eval_generation_steps 200 \
    --save_steps 500 \
    --generated_audio_save_dir "./${ROOT}/generated_audios_emotion" \
    --post_training_generation_eval true \
    --compute_clap_similarity_metric true \
    --compute_noise_level_metric true \
    --compute_ser_metric true \
    --ser_model_name_or_path "/work/tc068/tc068/tianqi30/.cache/modelscope/models/iic/emotion2vec_plus_large" \
    --noise_level_to_compute_clean_wer 25 \
    --add_audio_samples_to_wandb false \
    --asr_model_name_or_path "distil-whisper/distil-large-v2" \
    --clap_model_name_or_path "laion/larger_clap_music_and_speech"
```
