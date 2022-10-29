# Adversarial training with adapter in NLP/NLU

Adversarial training and adapter structure benefit pre-trained model transfer learning, especially in low-resource datasets. Here we upload different kinds of adversarial training algorithms w/o adapters for easily modify. *A_adapter* includes several adversarial algorithms with adapter pre-trained models. *Adversarial Training for NLU* includes the same but without the adapter.

## Instructions
This work is based on [huggingface/transformers](https://github.com/huggingface/transformers) and [adapter-Hub](https://github.com/adapter-hub/adapter-transformers).
Notice that w/o adapter does **NOT** share the same environment. It would be best if builts them separately.

1. Set up
```
$ pip install -r requirements.txt
```
2. Training
```
export TASK_NAME=cola
python run_glue_Aadapter.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 7e-4 \
  --num_train_epochs 10
```
3. Using other pre-trained model
If you want to change for other pre-trained like RoBERTa, don't forget to give the **base_model** name to the adversarial training class.
```
# run_glue_XXX.py

# init adversarial class
adv = Aadapter(adv_K=3, adv_lr=1e-1, adv_init_mag=2e-2, adv_max_norm=1.0, adv_norm_type='l2', base_model='roberta')
```

### computation cost
Memory:

We use single GPU(NVIDIA GeForce RTX 3090) and 20~22G RAM in our setting. The batch size and sentence length depend on your device.

Time:
|   Task   |  Metric  | Training time |
| -------- | -------- | ------------- |
|   CoLA   | Matthews corr. | 3 min |
|   SST-2  | Accuracy | 30 min |
|   MRPC   | F1/Accuracy | 3 min |
|   STS-B  | Pearson/Spearman corr. | 4 min |
|   QQP    | Accuracy/F1 | 4 hr 21 min |
|   MNLI   | Matched acc./Mismatched acc. | 5 hr 46 min |
|   QNLI   | Accuracy | 1 hr 40 min |
|   RTE    | Accuracy | 3 min |
|   WNLI   | Accuracy | 30 sec |


## Automatic Speech Recognition(ASR)
We also try to extend our work to ASR pre-trained models, such as Wave2vec2.0 and HuBERT.
In ASR, please use the same environment with *Adversarial Training for NLU*, please install requirements.txt and replace transformers/trainer.py with our trainer.py.

1. Librispeech

HuBERT
```
python run_speech_recognition_ctc.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="facebook/hubert-large-ll60k" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./hubert-large-ll60k-librispeech-clean-100h-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="3" \
	--per_device_train_batch_size="4" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--fp16 \
	--group_by_length \
	--do_train --do_eval
```
Wave2vec 2.0
```
python run_speech_recognition_ctc.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="facebook/wav2vec2-large-lv60" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./wav2vec2-librispeech-clean-100h-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="3" \
	--per_device_train_batch_size="4" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--fp16 \
	--group_by_length \
	--do_train --do_eval
```
2. Common Voice

HuBERT
```
python run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/hubert-large-ll60k" \
	--dataset_config_name="tr" \
	--output_dir="./hubert-common_voice-tr-demo" \
	--overwrite_output_dir \
	--num_train_epochs="20" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--length_column_name="input_length" \
	--save_steps="400" \
	--eval_steps="100" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--do_train --do_eval 
```
Wave2vec2.0
```
python run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo" \
	--overwrite_output_dir \
	--num_train_epochs="20" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--length_column_name="input_length" \
	--save_steps="400" \
	--eval_steps="100" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--do_train --do_eval 
```

### Computation cost
Memory:

We use single GPU(NVIDIA GeForce RTX 3090) and 20~22G RAM in our setting. The batch size depends on your device.

Time:
|   Task   | Training time |
| -------- |------------- |
|   CommonVoice-HuBERT   | 2hr 8min |
|   CommonVoice-Wav2vec2.0  | 18hr |
|   LibriSpeech-HuBERT   | 2hr 6min |
|   LibriSpeech-Wav2vec2.0  | 17hr 57min |
