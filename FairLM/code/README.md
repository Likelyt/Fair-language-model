# Code Base

## 1. Set Up
Activate Environment
```bash
conda activate FLM
```

## 2. Preprocessing
process sensitive descriptors, check 3 kingdoms (age, gender, race) with 16 classes.

```bash
cd preprocess\
```

```python
python data_preprocessing.ipynb
```

## 3. Train Baseline LM 
Choose portion of sample size to retrain the language model. If SUB_SAMPLE=10, it represents 1/10 percentage sample are used to retrain the Language model.
```bash
export SUB_SAMPLE=10
```

### 3.1 Train Baseline LM with OWTC
Retrain the language model GPT2 with OWTC dataset.

```bash
export SUB_SAMPLE=1
bash run_lm.sh
```

### 3.2 Train Baseline LM with OWTC with Masked Words
Retrain the language model GPT2 with masked sensitive words.

#### 3.2.1 with masked probability 1

Retrain the language model GPT2 with masked sensitive words with probability 1.
```bash
export MASK_PROB=1
bash run_lm_fair_mask.sh
```

#### 3.2.2 with masked probability 0.5
Retrain the language model GPT2 with masked sensitive words with probability 0.5.
```bash
export MASK_PROB=0.5
bash run_lm_fair_mask.sh
```

### 3.3 Load checkpoint to evaluate LM

```bash
bash run_lm_fair_mask_eval.sh
```

## 4. Train Classifier
Choose different sample size to train the classifier
```bash
export subsample_size = 1,000,000 or 8,000,000
```

### 4.1 Train GPT2 Classifier
Ger the classifier of GPT2 to predict sensitive class.

```bash
export MODEL_SELECT='gpt2'
bash run_classifier.sh
```

### 4.2 Train RoBERTa Classifier
Ger the classifier of RoBERTa to predict sensitive class.

```bash
export MODEL_SELECT='roberta'
bash run_classifier.sh
```

## 5. Training Fair Language Model
Train the FairLM model with generative adversarial training method with different sample size 10,000, 20,000, and 50,000.
```bash
export SUB_SAMPLE_SIZE=10000 #20000/50000
```

### 5.1 GPT2 - GPT2 initialized
Training a combination of generator (GPT2) - discriminator (GPT2 - initialized)

```bash
export MODEL_AUX='gpt2-aux'
export LOAD_PRETRAINED_CLS='init'
bash run_main_adv.sh
```

### 5.2 GPT2 - RoBERTa initialized
Training a combination of generator (GPT2) - discriminator (RoBERTa - initialized)

```bash
export MODEL_AUX='roberta-base-aux'
export LOAD_PRETRAINED_CLS='init'
bash run_main_adv.sh
```

### 5.3 GPT2 - GPT2 pretrained
Training a combination of generator (GPT2) - discriminator (GPT2 - pretrained)

```bash
export MODEL_AUX='gpt2-aux'
export LOAD_PRETRAINED_CLS='pretrain'
bash run_main_adv.sh
```

### 5.4 GPT2 - GPT2 pretrained
Training a combination of generator (GPT2) - discriminator (RoBERTa - pretrained)

```bash
export MODEL_AUX='roberta-base-aux'
export LOAD_PRETRAINED_CLS='pretrain'
bash run_main_adv.sh
```

All results will be collected in 3 different sample size 10,000, 20,000, and 50,000.


## 6. Run Evaluation
Evaluate different FairLM models with different training sample size 10,000, 20,000, and 50,000.
```bash
export SUB_SAMPLE_SIZE=10000 #20000/50000
```

### 6.1 Run Evaluation on GPT2 - GPT2 initialized
Evaluate  GPT2 - GPT2 (initialized) model fairness performance on the age, gender, and race prompts.

```bash
export LOAD_PRETRAINED_CLS='intialized' 
export MODEL_AUX='gpt2-base'
bash run_eval.sh
```

### 6.2 Run Evaluation on GPT2 - RoBERTa initialized
Evaluate  GPT2 - RoBERTa (initialized) model fairness performance on the age, gender, and race prompts.

```bash
export LOAD_PRETRAINED_CLS='intialized' 
export MODEL_AUX='roberta-base'
bash run_eval.sh
```

### 6.3 Run Evaluation on GPT2 - GPT2 pretrained

Evaluate  GPT2 - GPT2 (pretrained) model fairness performance on the age, gender, and race prompts.

```bash
export LOAD_PRETRAINED_CLS='pretrained' 
export MODEL_AUX='gpt2-base'
bash run_eval.sh
```

### 6.4 Run Evaluation on GPT2 - RoBERTa pretrained

Evaluate  GPT2 - RoBERTa (pretrained) model fairness performance on the age, gender, and race prompts.

```bash
export LOAD_PRETRAINED_CLS='pretrained' 
export MODEL_AUX='roberta-base'
bash run_eval.sh
```

## 7. Run Plot Loss

Plot the loss of the FairLM and analysis the best hyperparameter combination such as the DIS-GEN update ratio, alpha, sample size, and final result including perpleixty and fairness result.
```
RUN plot_train.ipynb 
```

## 8. Extra

```bash
cd debiaswe/
```
debiaswe folder contain a bias caluation package and related code from github (https://github.com/tolga-b/debiaswe), which is used for the evaluation of gender bias.

## 9. Ablation Study

1. In section 5, we can also test different discrimator-generator update ratio to select the best model.

2. In section 5, we can also test whether sample size affect the performance for example, 50K (4 epochs) vs 100K (2 epochs) vs 200K (1 epoch).




