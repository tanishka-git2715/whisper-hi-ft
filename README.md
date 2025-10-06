# whisper-hi-ft

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small)

It achieves the following results on the evaluation set:
- Loss: 1.0656
- Wer: 2.9786

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 12   | 1.2625          |
| No log        | 2.0   | 24   | 1.0992          |
| No log        | 3.0   | 36   | 1.0656          |


### Framework versions

- Transformers 4.56.1
- Pytorch 2.8.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.0
