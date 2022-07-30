# Adversarial training with adapter in NLP/NLU

Adversarial training and adapter structure benefit pre-trained model transfer learning, especially in low-resource datasets. Here we upload different kinds of adversarial training algorithms w/o adapters for easily modify. A_adapter includes several adversarial algorithms with adapter pre-trained models. Adversarial Training for NLU includes the same but without the adapter.

## Instructions
This work is based on huggingface transformers and adapter-Hub.
Notice that w/o adapter does *NOT* share the same environment. It would be best if builts them separately.

1. Set up
```
$ pip install -r requirements.txt
```
2. Training
```
export TASK_NAME=cola
python run_glue_ADAS.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 7e-4 \
  --num_train_epochs 10
```
