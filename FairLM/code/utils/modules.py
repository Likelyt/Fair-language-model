import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

class NetLM(nn.Module):
    """
    Network added after the Language Model
    """
    def __init__(self, input_emb_size, output_emb_size):
        super().__init__()
        self.linear_key = nn.Linear(input_emb_size, output_emb_size, bias = False)
        self.linear_value = nn.Linear(input_emb_size, output_emb_size, bias = False)

    def forward(self, x, name):
        if name == 'key':
            x = F.relu(self.linear_key(x))
        elif name == 'value':
            x = F.relu(self.linear_value(x))
        return x
    
class NetAdv(nn.Module):
    """
    Network added after the classification network
    """
    def __init__(self, input_emb_size, output_emb_size):
        super().__init__()
        self.linear_query = nn.Linear(input_emb_size, output_emb_size, bias = False)
        self.linear_concat = nn.Linear(output_emb_size * 2, output_emb_size, bias = False)

    def forward(self, x, name):
        if name == 'query':
            x = F.relu(self.linear_query(x))    
        elif name == 'concat':
            x = F.relu(self.linear_concat(x))

        return x 


def hyper_parameters(model, dataloader, epochs, lr = 2e-5, eps=1e-8):
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # default is 1e-8.
                      )
    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return optimizer, total_steps, scheduler   
    
# dis: aux, (net1, net2)
# gen: main, (net1, net2)
def hyper_parameters_net(net1, net2, dataloader, epochs, lr = 2e-5, eps=1e-8):
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    params = list(net1.parameters()) + list(net2.parameters())
    optimizer = AdamW(params, lr = 2e-5, eps = 1e-8)
    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return optimizer, total_steps, scheduler
