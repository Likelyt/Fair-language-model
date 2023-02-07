import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm, trange
from utils.util import NetSwitchSchedule, NetSwitchScheduleNew, CalculateAttention, classification_loss, save_checkpoint_model,save_loss_plot
from utils.data_processing import Batches_gpt2, Batches_Bert


logger = logging.getLogger(__name__)


def model_train_gpt2(dataloader, model_main, model_aux, net1, net2,
                tokenizer_main, tokenizer_aux,
                optimizer_main, scheduler_main, optimizer_aux, scheduler_aux, optimizer_net, scheduler_net, 
                switch_freq, alpha, block_size, n_gpu, 
                device, class2kingdom, opt_method,
                output_main_dir, output_aux_dir, output_net1_dir, output_net2_dir,
                epoch, args):
    """
    Train GPT2-GPT2 model
    """
    ##########################################################################################
    #############                         Training functions                     #############
    ##########################################################################################
    # Tracking variables.
    predictions_labels_adv = []
    predictions_labels_cls = []
    true_labels = []
    
    # Total loss for this epoch.
    # Discriminator Loss
    total_cls_loss = []
    # Generator Loss
    total_adv_loss = []
    Generator_classification_loss = []
    Generator_lm_loss = []
    
    # Put the model into training mode.
    model_main.train()
    model_aux.train()
    net1.train()
    net2.train()

    # Switch between discriminator training and generator training
    # discriminator_train_plan, generator_train_plan = NetSwitchSchedule(dataloader, switch_freq)
    discriminator_train_plan, generator_train_plan = NetSwitchScheduleNew(dataloader, args.switch_freq_dis, args.switch_freq_gen)
    count_cls_step = 0
    count_adv_step = 0
    
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        batches_lm, batches_aux = Batches_gpt2(batch, device)

        # Part I - Train the Discriminator
        if step in discriminator_train_plan:
            #print('step %d, training at Descriminator' % (step))    
            model_aux.zero_grad()
            net1.zero_grad()
            net2.zero_grad()

            # Part I-1 - Aux
            # print(batches_aux)
            # print(batches_aux['input_ids'].size())
            # print(batches_aux['attention_mask'].size())
            
            outputs_aux = model_aux(**batches_aux)
            loss_cls, logits_cls = outputs_aux[:2]

            # Step 2.1: laod hidden_emb_cls
            hidden_emb_cls = outputs_aux.hidden_states[12][:,-1,:]


            # Part I-2 - LM
            outputs_main = model_main(input_ids = batches_lm['input_ids'].detach(), 
                                      attention_mask = batches_lm['attention_mask'].detach(),
                                      labels = batches_lm['input_ids'].detach())
            # Prepare LM embedding
            fair_mask = batches_lm['fair_mask'].detach()
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = (torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1)).detach() # no gradient flow here

            # Part I-3 - Cross

            # Step 2.2: calculate cross_embedding
            # cross_embedding: batch_size * embedding_size
            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                    net1, net2, device, block_size) # needs gradient flow

            # Step 2.3: concatenate the cross_embedding with the self-embedding
            #           final_embedding: batch_size * embedding size
            # self_embedding = net2(hidden_emb_cls, 'query')
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            # step 2.4: predict the final logits
            #           output_aux: batch_size * categories
            model_aux = (model_aux.module if hasattr(model_aux, "module") else model_aux)
            mlp_prediction = model_aux.score(final_embedding)

            # step 2.5: calculate the classification loss      
            # Optional 1:          
            # loss_cls = classification_loss(mlp_prediction, batches_aux['labels'])
            
            # Optional 2:
            # remove those samples has no sensitive words, Label = 0
            s_label = []
            for i in range(len(batches_aux['labels'])):
                if batches_aux['labels'][i] != 0:
                    s_label.append(i)
            if len(s_label) != 0:
                loss_cls = classification_loss(mlp_prediction[s_label], batches_aux['labels'][s_label])
            else:
                loss_cls = torch.tensor(0.0).type(torch.float32).to(device)
                loss_cls.requires_grad=True


            if n_gpu > 1:
                loss_cls = loss_cls.mean()  # mean() to average on multi-gpu parallel training
                total_cls_loss.append(loss_cls.item())
            else:
                total_cls_loss.append(loss_cls.item())

            # Perform a backward pass to calculate the gradients.
            loss_cls.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model_aux.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_aux.step()
            optimizer_net.step()

            # Update the learning rate.
            scheduler_aux.step()
            scheduler_net.step()

            # Move logits and labels to CPU
            logits_cls = mlp_prediction.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            predictions_labels_cls += logits_cls.argmax(axis=-1).flatten().tolist()

            count_cls_step += 1
            if count_cls_step % 100 == 0: 
                logger.info('CLS step %d, Average Classification loss %.3f' , step + 1, sum(total_cls_loss)/len(total_cls_loss))

        #Part 2 - Train the Generator  
        elif step in generator_train_plan:

            model_main.zero_grad()
            net1.zero_grad()
            net2.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            outputs_main = model_main(input_ids = batches_lm['input_ids'],
                                      attention_mask = batches_lm['attention_mask'],
                                      labels=batches_lm['input_ids'])
            #################################################################################
            ##########     Step 1: extract the hidden embedding with fair_maks        #######
            #################################################################################

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            # logits: batch_size * seq_len * vocab_size
            # Calculate the language model loss
            loss_lm, logits_lm = outputs_main[:2]

            #################################################################################
            ##########     Step 2: Calculate the classification loss                  #######
            #################################################################################
            # outputs.hidden_states[-1]: batch_size * sequence_len * emb_size
            # fair_mask: batch_size * sequence_len
            fair_mask = batches_lm['fair_mask'].detach()
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1) 

            # Step 2.1: load cls_embedding from GPT-2
            # hidden_emb_cls: batch_size * embedding size

            # output the embedding from the auxilary network
            outputs_aux = model_aux(input_ids = batches_aux['input_ids'].detach(), attention_mask = batches_aux['attention_mask'].detach())
            # hidden_emb_cls: batch_size * emb_size
            hidden_emb_cls = outputs_aux.hidden_states[12][:,-1,:].detach()

            # Step 2.2: calculate cross_embedding
            # cross_embedding: batch_size * embedding_size/2
            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                    net1, net2, device, block_size) # needs gradient flow

            # Step 2.3: concatenate the cross_embedding with the self-embedding
            #           self_embedding: batch_size * embedding_size/2
            #self_embedding = net2(hidden_emb_cls, 'query') # require_grad=True, since we need to train net2.parameters
            #           final_embedding: batch_size * embedding size
            #final_embedding = torch.cat((cross_embedding, self_embedding), dim = 1)
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            # step 2.4: predict the final logits
            #           output_aux: batch_size * categories
            model_aux = (model_aux.module if hasattr(model_aux, "module") else model_aux)
            mlp_prediction = model_aux.score(final_embedding)

            # step 2.5: calculate the classification loss
            # Optional 1:          
            #loss_adv = classification_loss(mlp_prediction, batches_aux['labels'])
            # Optional 2:
            # remove those samples has no sensitive words
            s_label = []
            for i in range(len(batches_aux['labels'])):
                if batches_aux['labels'][i] != 0:
                    s_label.append(i)
            #logger.info("Non Zero Position %s", s_label)
            if len(s_label) != 0:
                loss_adv = classification_loss(mlp_prediction[s_label], batches_aux['labels'][s_label])
            else:
                loss_adv = torch.tensor(0.0).type(torch.float32).to(device)
                loss_adv.requires_grad=True

            

            #################################################################################
            ##########     Step 3: Calculate the total loss                           #######
            #################################################################################
            # calculate the sum loss of language loss(loss_LM) - classificaiton loss (loss_cls)
            total_loss = loss_lm - alpha * loss_adv

            if n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
                Generator_classification_loss.append(loss_adv.item())
                Generator_lm_loss.append(loss_lm.mean().item())
                total_adv_loss.append(total_loss.item())
            else:
                Generator_classification_loss.append(loss_adv.item())
                Generator_lm_loss.append(loss_lm.mean().item())
                total_adv_loss.append(total_loss.item())

            # Perform a backward pass to calculate the gradients.
            total_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model_main.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_main.step()
            if opt_method == 'method_multi_opt':
                optimizer_net.step()

            # Update the learning rate.
            scheduler_main.step()
            if opt_method == 'method_multi_opt':
                scheduler_net.step()

            # Move logits and labels to CPU
            total_loss = total_loss.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            #predictions_labels_cls += logits_cls.argmax(axis=-1).flatten().tolist()
            count_adv_step += 1
            if count_adv_step % 100 == 0: 
                logger.info('Adv Step %d, LM loss %.3f; Average Adversarial loss %.3f', step + 1, loss_lm.mean().item(), sum(total_adv_loss)/len(total_adv_loss))

    # Calculate the average loss over the training data.
    avg_cls_epoch_loss = sum(total_cls_loss) / count_cls_step
    avg_adv_epoch_loss = sum(total_adv_loss) / count_adv_step    
    
    # Saving model
    save_checkpoint_model(model_main, output_main_dir, 'main', epoch, alpha, tokenizer_main)
    save_checkpoint_model(model_aux, output_aux_dir, 'aux', epoch, alpha, tokenizer_aux)
    save_checkpoint_model(net1, output_net1_dir, 'net1', epoch, alpha)
    save_checkpoint_model(net2, output_net2_dir, 'net2', epoch, alpha)
    
    save_loss_plot(args, epoch, total_cls_loss, Generator_lm_loss, Generator_classification_loss, total_adv_loss)

    return avg_cls_epoch_loss, avg_adv_epoch_loss




def model_validation_gpt2(model_main, model_aux, net1, net2, 
                    dataloader, eval_output_dir, block_size, 
                    alpha, opt_method, batch_size, 
                    device, epoch, prefix=""):
    """
    Perform validation on the validation set for GPT2-GPT2 model
    """
    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model_main.eval()
    model_aux.eval()
    net1.eval()
    net2.eval()

    eval_lm_loss = 0
    eval_adv_loss = 0
    eval_total_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()
        # move batch to device
        batches_lm, batches_aux = Batches_gpt2(batch, device)
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            # 1. Loss LM
            outputs_main = model_main(input_ids = batches_lm['input_ids'],
                                       attention_mask = batches_lm['attention_mask'],
                                       labels=batches_lm['input_ids'])
            loss_lm, logits_lm = outputs_main[:2]
            eval_lm_loss += loss_lm.mean().item()

            # 2. Loss Adv
            fair_mask = batches_lm['fair_mask']
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1) 


            # output the embedding from the auxilary network
            outputs_aux = model_aux(input_ids = batches_aux['input_ids'], attention_mask = batches_aux['attention_mask'])
            # hidden_emb_cls: batch_size * emb_size
            hidden_emb_cls = outputs_aux.hidden_states[12][:,-1,:]

            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                net1, net2, device, block_size) # needs gradient flow

            #self_embedding = net2(hidden_emb_cls, 'query')
            #final_embedding = torch.cat((cross_embedding, self_embedding), dim = 1)
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            model_aux = (model_aux.module if hasattr(model_aux, "module") else model_aux)
            mlp_prediction = model_aux.score(final_embedding)
            #loss_adv = classification_loss(mlp_prediction, batches_aux['labels'])
            s_label = []
            for i in range(len(batches_aux['labels'])):
                if batches_aux['labels'][i] != 0:
                    s_label.append(i)
            if len(s_label) != 0:
                loss_adv = classification_loss(mlp_prediction[s_label], batches_aux['labels'][s_label])
            else:
                loss_adv = torch.tensor(0.0).type(torch.float32).to(device)
            eval_adv_loss += loss_adv.mean().item()

            # total_loss
            eval_total_loss = eval_lm_loss - alpha * eval_adv_loss

        nb_eval_steps += 1    

    eval_lm_loss = eval_lm_loss / nb_eval_steps
    eval_adv_loss = eval_adv_loss / nb_eval_steps
    eval_total_loss = eval_total_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_lm_loss))

    result = {"perplexity": perplexity, 'total_loss': eval_total_loss, 'lm_loss': eval_lm_loss, 'adv_loss': eval_adv_loss}
    
    eval_evaluation_dir = os.path.join(eval_output_dir, prefix, 'checkpoint-%s'%str(epoch))
    eval_evaluation_dir_file = os.path.join(eval_output_dir, prefix, 'checkpoint-%s'%str(epoch), 
                                        "eval_results_B_%s_M_%s_A_%s.txt" % (batch_size, opt_method, str(alpha)))
    if not os.path.exists(eval_evaluation_dir):
        os.makedirs(eval_evaluation_dir)
        
    with open(eval_evaluation_dir_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        #logger.info("***** Eval results {} *****", prefix)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            
    return result


def model_train_bert(dataloader, model_main, model_aux, net1, net2,
                tokenizer_main, tokenizer_aux,
                optimizer_main, scheduler_main, optimizer_aux, scheduler_aux, optimizer_net, scheduler_net, 
                switch_freq, alpha, block_size, n_gpu, 
                device, class2kingdom, opt_method, CLS_SELECTION_METHOD,
                output_main_dir, output_aux_dir, output_net1_dir, output_net2_dir,
                epoch, args):
    """
    Model training for GPT2-RoBERTa model
    """
    ##########################################################################################
    #############                         Training functions                     #############
    ##########################################################################################
    # Tracking variables.
    predictions_labels_adv = []
    predictions_labels_cls = []
    true_labels = []
    
    # Total loss for this epoch.
    # Discriminator Loss
    total_cls_loss = []
    # Generator Loss
    total_adv_loss = []
    Generator_classification_loss = []
    Generator_lm_loss = []
    
    # Put the model into training mode.
    model_main.train()
    model_aux.train()
    net1.train()
    net2.train()

    # Switch between discriminator training and generator training
    # discriminator_train_plan, generator_train_plan = NetSwitchSchedule(dataloader, switch_freq)
    discriminator_train_plan, generator_train_plan = NetSwitchScheduleNew(dataloader, args.switch_freq_dis, args.switch_freq_gen)

    count_cls_step = 0
    count_adv_step = 0
    
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels_cls'].numpy().flatten().tolist()

        batches_lm, batches_aux = Batches_Bert(batch, device)

        # Part I - Train the Discriminator
        if step in discriminator_train_plan:
            #print('step %d, training at Descriminator' % (step))    
            model_aux.zero_grad()
            net1.zero_grad()
            net2.zero_grad()

            # Part I-1 - Aux
            #outputs_aux = model_aux(**batches_aux)
            outputs_aux = model_aux(input_ids = batches_aux['input_ids_cls'], \
                                    attention_mask = batches_aux['attention_mask_cls'], \
                                    labels = batches_aux['labels_cls'])
            
            ################################################################################################
            ########            Method Combination or [CLS]_TOKEN                                    ########
            ################################################################################################
            
            
            if CLS_SELECTION_METHOD == '[CLS]_TOKEN':
                loss_cls, logits_cls = outputs_aux[:2]
                # Step 2.1: laod hidden_emb_cls
                hidden_emb_cls = outputs_aux.hidden_states[-1][:,0,:]


            # Part I-2 - LM
            outputs_main = model_main(input_ids = batches_lm['input_ids_gen'].detach(), 
                                      attention_mask = batches_lm['attention_mask_gen'].detach(),
                                      labels = batches_lm['input_ids_gen'].detach())
            # Prepare LM embedding
            fair_mask = batches_lm['fair_mask_gen'].detach()
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = (torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1)).detach() # no gradient flow here

            # Part I-3 - Cross

            # Step 2.2: calculate cross_embedding
            # cross_embedding: batch_size * embedding_size
            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                    net1, net2, device, block_size) # needs gradient flow

            # Step 2.3: concatenate the cross_embedding with the self-embedding
            #           final_embedding: batch_size * embedding size
            # self_embedding = net2(hidden_emb_cls, 'query')
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            # step 2.4: predict the final logits
            #           output_aux: batch_size * categories
            #mlp_prediction = model_aux.module.score(final_embedding)
            if n_gpu == 1:
                mlp_prediction = model_aux.classifier(final_embedding.unsqueeze(1))
            elif n_gpu > 1:
                mlp_prediction = model_aux.module.classifier(final_embedding.unsqueeze(1))

            # step 2.5: calculate the classification loss
            # loss_cls = classification_loss(mlp_prediction, batches_aux['labels_cls'])
            # Optional 2:
            # remove those samples has no sensitive words, Label = 0
            s_label = []
            for i in range(len(batches_aux['labels_cls'])):
                if batches_aux['labels_cls'][i] != 0:
                    s_label.append(i)
            if len(s_label) != 0:
                loss_cls = classification_loss(mlp_prediction[s_label], batches_aux['labels_cls'][s_label])
            else:
                loss_cls = torch.tensor(0.0).type(torch.float32).to(device)
                loss_cls.requires_grad=True


            if n_gpu > 1:
                loss_cls = loss_cls.mean()  # mean() to average on multi-gpu parallel training
                total_cls_loss.append(loss_cls.item())
            else:
                total_cls_loss.append(loss_cls.item())

            # Perform a backward pass to calculate the gradients.
            loss_cls.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model_aux.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_aux.step()
            optimizer_net.step()

            # Update the learning rate.
            scheduler_aux.step()
            scheduler_net.step()

            # Move logits and labels to CPU
            logits_cls = mlp_prediction.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            predictions_labels_cls += logits_cls.argmax(axis=-1).flatten().tolist()

            count_cls_step += 1
            if count_cls_step % 5 == 0: 
                logger.info('CLS step %d, Average Classification loss %.3f' , step + 1, sum(total_cls_loss)/len(total_cls_loss))

        #Part 2 - Train the Generator  
        elif step in generator_train_plan:

            model_main.zero_grad()
            net1.zero_grad()
            net2.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            outputs_main = model_main(input_ids = batches_lm['input_ids_gen'],
                                      attention_mask = batches_lm['attention_mask_gen'],
                                      labels=batches_lm['input_ids_gen'])
            #################################################################################
            ##########     Step 1: extract the hidden embedding with fair_maks        #######
            #################################################################################

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            # logits: batch_size * seq_len * vocab_size
            # Calculate the language model loss
            loss_lm, logits_lm = outputs_main[:2]

            #################################################################################
            ##########     Step 2: Calculate the classification loss                  #######
            #################################################################################
            # outputs.hidden_states[-1]: batch_size * sequence_len * emb_size
            # fair_mask: batch_size * sequence_len
            fair_mask = batches_lm['fair_mask_gen'].detach()
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1) 

            # Step 2.1: load cls_embedding from GPT-2
            # hidden_emb_cls: batch_size * embedding size
            
            if CLS_SELECTION_METHOD == '[CLS]_TOKEN':
                # output the embedding from the auxilary network
                outputs_aux = model_aux(input_ids = batches_aux['input_ids_cls'].detach(), \
                                        attention_mask = batches_aux['attention_mask_cls'].detach())
                # hidden_emb_cls: batch_size * emb_size
                hidden_emb_cls = outputs_aux.hidden_states[-1][:,0,:].detach()

            # Step 2.2: calculate cross_embedding
            # cross_embedding: batch_size * embedding_size/2
            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                    net1, net2, device, block_size) # needs gradient flow

            # Step 2.3: concatenate the cross_embedding with the self-embedding
            #           self_embedding: batch_size * embedding_size/2
            #self_embedding = net2(hidden_emb_cls, 'query') # require_grad=True, since we need to train net2.parameters
            #           final_embedding: batch_size * embedding size
            #final_embedding = torch.cat((cross_embedding, self_embedding), dim = 1)
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            # step 2.4: predict the final logits
            #           output_aux: batch_size * categories
            #mlp_prediction = model_aux.module.score(final_embedding)
            if n_gpu == 1:
                mlp_prediction = model_aux.classifier(final_embedding.unsqueeze(1))
            elif n_gpu > 1:
                mlp_prediction = model_aux.module.classifier(final_embedding.unsqueeze(1))

            # step 2.5: calculate the classification loss
            # option 1:
            #loss_adv = classification_loss(mlp_prediction, batches_aux['labels_cls'])
            # Option 2:
            # remove those samples has no sensitive words
            s_label = []
            for i in range(len(batches_aux['labels_cls'])):
                if batches_aux['labels_cls'][i] != 0:
                    s_label.append(i)
            #logger.info("Non Zero Position %s", s_label)
            if len(s_label) != 0:
                loss_adv = classification_loss(mlp_prediction[s_label], batches_aux['labels_cls'][s_label])
            else:
                loss_adv = torch.tensor(0.0).type(torch.float32).to(device)
                loss_adv.requires_grad=True

            #################################################################################
            ##########     Step 3: Calculate the total loss                           #######
            #################################################################################
            # calculate the sum loss of language loss(loss_LM) - classificaiton loss (loss_cls)
            total_loss = loss_lm - alpha * loss_adv

            if n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
                Generator_classification_loss.append(loss_adv.item())
                Generator_lm_loss.append(loss_lm.mean().item())
                total_adv_loss.append(total_loss.item())
            else:
                Generator_classification_loss.append(loss_adv.item())
                Generator_lm_loss.append(loss_lm.mean().item())
                total_adv_loss.append(total_loss.item())

            # Perform a backward pass to calculate the gradients.
            total_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model_main.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer_main.step()
            if opt_method == 'method_multi_opt':
                optimizer_net.step()

            # Update the learning rate.
            scheduler_main.step()
            if opt_method == 'method_multi_opt':
                scheduler_net.step()

            # Move logits and labels to CPU
            total_loss = total_loss.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            #predictions_labels_cls += logits_cls.argmax(axis=-1).flatten().tolist()
            count_adv_step += 1
            if count_adv_step % 5 == 0: 
                logger.info('Adv Step %d, LM loss %.3f; Average Adversarial loss %.3f', step + 1, loss_lm.mean().item(), sum(total_adv_loss)/len(total_adv_loss))

    # Calculate the average loss over the training data.
    avg_cls_epoch_loss = sum(total_cls_loss) / count_cls_step
    avg_adv_epoch_loss = sum(total_adv_loss) / count_adv_step    
    
    # Saving model
    save_checkpoint_model(model_main, output_main_dir, 'main', epoch, alpha, tokenizer_main)
    save_checkpoint_model(model_aux, output_aux_dir, 'aux', epoch, alpha, tokenizer_aux)
    save_checkpoint_model(net1, output_net1_dir, 'net1', epoch, alpha)
    save_checkpoint_model(net2, output_net2_dir, 'net2', epoch, alpha)
    
    save_loss_plot(args, epoch, total_cls_loss, Generator_lm_loss, Generator_classification_loss, total_adv_loss)

    return avg_cls_epoch_loss, avg_adv_epoch_loss


def model_validation_bert(model_main, model_aux, net1, net2, 
                    dataloader, eval_output_dir, block_size, 
                    alpha, opt_method, CLS_SELECTION_METHOD, batch_size, 
                    device, epoch, prefix=""):
    """
    Perform validation on the validation set for GPT2-Roberta Model
    """
    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model_main.eval()
    model_aux.eval()
    net1.eval()
    net2.eval()

    eval_lm_loss = 0
    eval_adv_loss = 0
    eval_total_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels_cls'].numpy().flatten().tolist()
        # move batch to device
        batches_lm, batches_aux = Batches_Bert(batch, device)
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            # 1. Loss LM
            outputs_main = model_main(input_ids = batches_lm['input_ids_gen'],
                                       attention_mask = batches_lm['attention_mask_gen'],
                                       labels=batches_lm['input_ids_gen'])
            loss_lm, logits_lm = outputs_main[:2]
            eval_lm_loss += loss_lm.mean().item()

            # 2. Loss Adv
            fair_mask = batches_lm['fair_mask_gen']
            # hidden_emb_lm: batch_size * emb_dim * sequence_len
            hidden_emb_lm = torch.transpose(outputs_main.hidden_states[-1],1,2) * fair_mask.unsqueeze(1) 
            
            if CLS_SELECTION_METHOD == '[CLS]_TOKEN':
                # output the embedding from the auxilary network
                outputs_aux = model_aux(input_ids = batches_aux['input_ids_cls'], \
                                        attention_mask = batches_aux['attention_mask_cls'])
                # hidden_emb_cls: batch_size * emb_size
                hidden_emb_cls = outputs_aux.hidden_states[-1][:,0,:]

            cross_embedding = CalculateAttention(hidden_emb_lm, hidden_emb_cls, 
                                                net1, net2, device, block_size) # needs gradient flow

            #self_embedding = net2(hidden_emb_cls, 'query')
            #final_embedding = torch.cat((cross_embedding, self_embedding), dim = 1)
            concat_embedding = torch.cat((cross_embedding, hidden_emb_cls), dim = 1) # 768 + 768 = 1536
            final_embedding = net2(concat_embedding, 'concat') # 768 * 2 -> 768

            model_aux = (model_aux.module if hasattr(model_aux, "module") else model_aux)
            mlp_prediction = model_aux.classifier(final_embedding.unsqueeze(1))
            #loss_adv = classification_loss(mlp_prediction, batches_aux['labels_cls'])
            s_label = []
            for i in range(len(batches_aux['labels_cls'])):
                if batches_aux['labels_cls'][i] != 0:
                    s_label.append(i)
            if len(s_label) != 0:
                loss_adv = classification_loss(mlp_prediction[s_label], batches_aux['labels_cls'][s_label])
            else:
                loss_adv = torch.tensor(0.0).type(torch.float32).to(device)

            eval_adv_loss += loss_adv.mean().item()

            # total_loss
            eval_total_loss = eval_lm_loss - alpha * eval_adv_loss

        nb_eval_steps += 1    

    eval_lm_loss = eval_lm_loss / nb_eval_steps
    eval_adv_loss = eval_adv_loss / nb_eval_steps
    eval_total_loss = eval_total_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_lm_loss))

    result = {"perplexity": perplexity, 'total_loss': eval_total_loss, 'lm_loss': eval_lm_loss, 'adv_loss': eval_adv_loss}
    
    eval_evaluation_dir = os.path.join(eval_output_dir, prefix, 'checkpoint-%s'%str(epoch))
    eval_evaluation_dir_file = os.path.join(eval_output_dir, prefix, 'checkpoint-%s'%str(epoch), 
                                        "eval_results_B_%s_M_%s_A_%s.txt" % (batch_size, opt_method, str(alpha)))
    if not os.path.exists(eval_evaluation_dir):
        os.makedirs(eval_evaluation_dir)
    
    with open(eval_evaluation_dir_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        #logger.info("***** Eval results {} *****", prefix)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            
    return result