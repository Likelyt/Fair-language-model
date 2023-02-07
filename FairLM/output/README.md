# Checkpoint Directory

## 1. Pretrained model checkpoint directory

```bash
cd main/pretrained
```

```
pretrained
│
└───lm/block-256                    # Language model with maximum length 256
|   |   fair_maks/                  # Language model training with masked sensitive words
│   │       └───mask_1              # Masked probability 1
│   │       └───mask_0.5            # Masked probability 0.5
│   │       └───mask_0.1            # Masked probability 0.1
│   │   regular/                    # Language model training without mask tokens 
│
└───cls/block-256/                  # classifier
│   │    gpt2-small/                # GPT2-small classifier
│   │    roberta-base-small/        # RoBERTa-small classifier
```


## 2. The Generator (Main) checkpoint's saving directory

```
main
│
└───gpt2-base                       # Generator model Language model 
|   |   8M-10000                    # Training sample size is 10,000
|   |      └───intialized           # Generator checkpoint with Discriminator is initiliazed
|   |      └───pretrained           # Generator checkpoint with Discriminator is pretrained
|   |   8M-20000                    # Training sample size is 20,000
|   |      └───intialized           # Generator checkpoint with Discriminator is initiliazed
|   |      └───pretrained           # Generator checkpoint with Discriminator is pretrained
|   |   8M-50000                    # Training sample size is 50,000
|   |      └───intialized           # Generator checkpoint with Discriminator is initiliazed
|   |      └───pretrained           # Generator checkpoint with Discriminator is pretrained
│
└───gpt2-main-gpt2-aux              # perpleixty result
└───gpt2-main-roberta-base-aux      # perpleixty result
```


## 3. The Discriminator (Aux) checkpoint's saving directory

```
aux
│
└───gpt2-aux-base                   # Discriminator model is gpt2
|   |   8M-10000                    # Training sample size is 10,000
|   |   8M-20000                    # Training sample size is 20,000
|   |   8M-50000                    # Training sample size is 50,000
└───roberta-base-aux-base           # Discriminator model is roberta
|   |   8M-10000                    # Training sample size is 10,000
|   |   8M-20000                    # Training sample size is 20,000
|   |   8M-50000                    # Training sample size is 50,000
```