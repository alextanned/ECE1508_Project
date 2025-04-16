import time
import torch
import torch.utils.data
import argparse
import os
import datetime
import wandb
import json
import nlpaug.augmenter.word as naw
import numpy as np
import copy

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F

########################################
# Data Preparation & Augmentation
########################################

def data_prep(args):
    # Load train and test data from JSON files.
    with open("train.json", "r", encoding="utf8") as f:
        train_data = json.load(f)
    with open("test.json", "r", encoding="utf8") as f:
        val_data = json.load(f)

    # If augmentation is requested, add augmented copies of the samples instead of replacing.
    if args.aug:
        aug_syn = naw.SynonymAug(aug_src='wordnet')
        aug_spell = naw.SpellingAug()
        new_train_samples = []
        for sample in train_data:
            # Create deep copies for two augmentation types.
            sample_syn = copy.deepcopy(sample)
            sample_spell = copy.deepcopy(sample)
            
            # Augment question and each option with synonym augmentation.
            sample_syn['query'] = aug_syn.augment(sample_syn['query'])[0]
            for key in sample_syn['options'].keys():
                sample_syn['options'][key] = aug_syn.augment(sample_syn['options'][key])[0]
            
            # Augment with spelling corrections.
            sample_spell['query'] = aug_spell.augment(sample_spell['query'])[0]
            for key in sample_spell['options'].keys():
                sample_spell['options'][key] = aug_spell.augment(sample_spell['options'][key])[0]
            
            new_train_samples.append(sample_syn)
            new_train_samples.append(sample_spell)
            
        train_data = train_data + new_train_samples

    if args.debug:
        train_data = train_data[:10]
        val_data = val_data[:10]

    # Determine the number of choices; assuming each sample's 'options' field is a dict.
    num_choices = len(train_data[0]['options'])
    return train_data, val_data, num_choices

########################################
# Utility Functions
########################################

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save-path',
        type=str,
        default="./",
        help='File path where the output files will be saved.'
    )

    # Here, we specify a default sentence-transformer (or any encoder) model.
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='The encoder model to be used (e.g. sentence-transformers/all-MiniLM-L6-v2, roberta-large, xlnet-large, etc.).'
    )

    parser.add_argument('--bs', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--aug', action='store_true', default=False, help='Apply data augmentation')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')

    args = parser.parse_args()
    return args

########################################
# Custom Model using Sentence Encoder & Cross-Attention
########################################

class CustomMultipleChoiceCrossAttentionModel(nn.Module):
    def __init__(self, encoder_name, hidden_size, dropout=0.1, num_heads=8):
        super(CustomMultipleChoiceCrossAttentionModel, self).__init__()
        # Load the encoder model (this is used for both question and options)
        self.encoder = AutoModel.from_pretrained(encoder_name, use_mems_eval=False)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

        # For the question, we use mean pooling over token embeddings.
        # For options, we want the full token sequence for cross-attention.
        # Define a multihead cross-attention layer.
        # Here, we use batch_first=True so that input shapes are (batch, seq, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                                      num_heads=num_heads,
                                                      dropout=dropout,
                                                      batch_first=True)
        # A classifier to go from the cross attention output (a fused vector) to a scalar logit.
        self.classifier = nn.Linear(hidden_size, 1)
    
    def mean_pooling(self, model_output, attention_mask):
        # model_output.last_hidden_state: (batch, seq_len, hidden_size)
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask  # (batch, hidden_size)
    
    def forward(self, q_input_ids, q_attention_mask, opt_input_ids, opt_attention_mask):
        """
        q_input_ids:      (batch, seq_len_q)
        q_attention_mask: (batch, seq_len_q)
        opt_input_ids:    (batch, num_choices, seq_len_opt)
        opt_attention_mask: (batch, num_choices, seq_len_opt)
        """
        batch_size, num_choices, seq_len_opt = opt_input_ids.size()
        
        # --- Encode the question ---
        q_outputs = self.encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        # Use mean pooling (or CLS token) to obtain a fixed vector.
        q_embed = self.mean_pooling(q_outputs, q_attention_mask)  # (batch, hidden_size)
        q_embed = self.dropout(q_embed)
        # We now have a fixed vector for each sample question.
        
        # --- Encode the options (without pooling) ---
        # Flatten the options to get shape (batch*num_choices, seq_len_opt)
        opt_input_ids_flat = opt_input_ids.view(batch_size * num_choices, seq_len_opt)
        opt_attention_mask_flat = opt_attention_mask.view(batch_size * num_choices, seq_len_opt)
        opt_outputs = self.encoder(input_ids=opt_input_ids_flat,
                                   attention_mask=opt_attention_mask_flat)
        # Get the sequence output (do not pool) -> (batch*num_choices, seq_len_opt, hidden_size)
        opt_sequence = opt_outputs.last_hidden_state
        
        # --- Cross-Attention ---
        # For each option, we want to use the corresponding question representation as query.
        # First, replicate the question embedding to align with options.
        # q_embed: (batch, hidden_size) => (batch, num_choices, hidden_size)
        q_rep = q_embed.unsqueeze(1).repeat(1, num_choices, 1)
        # Flatten to shape (batch*num_choices, 1, hidden_size)
        q_rep = q_rep.view(batch_size * num_choices, 1, self.hidden_size)
        
        # Prepare key_padding_mask for the options.
        # opt_attention_mask_flat is (batch*num_choices, seq_len_opt) with 1s for real tokens.
        # MultiheadAttention expects key_padding_mask of shape (batch, seq_len)
        key_padding_mask = (opt_attention_mask_flat == 0)  # bool: True where padding
        # Now apply cross attention: query = q_rep, key and value = opt_sequence.
        # q_rep shape: (batch*num_choices, 1, hidden_size)
        # opt_sequence shape: (batch*num_choices, seq_len_opt, hidden_size)
        attn_output, attn_weights = self.cross_attention(query=q_rep,
                                                         key=opt_sequence,
                                                         value=opt_sequence,
                                                         key_padding_mask=key_padding_mask)
        # attn_output shape: (batch*num_choices, 1, hidden_size)
        attn_output = attn_output.squeeze(1)  # (batch*num_choices, hidden_size)
        attn_output = self.dropout(attn_output)
        
        # Compute a logit from the cross-attention fused vector.
        logits = self.classifier(attn_output)  # (batch*num_choices, 1)
        logits = logits.view(batch_size, num_choices)  # (batch, num_choices)
        
        return logits

########################################
# Dataset Creation
########################################

def create_dataset(examples, tokenizer, num_choices):
    q_input_ids_list = []
    q_attention_mask_list = []
    opt_input_ids_list = []
    opt_attention_mask_list = []
    labels_list = []
    for example in examples:
        query = example['query']
        options = example['options']
        sorted_keys = sorted(options.keys())
        choices = [options[key] for key in sorted_keys]
        
        # Tokenize the question.
        tokenized_question = tokenizer(query, add_special_tokens=True,
                                       truncation=True, max_length=128, padding='max_length')
        # Tokenize each option independently.
        tokenized_options = [
            tokenizer(choice, add_special_tokens=True,
                      truncation=True, max_length=128, padding='max_length')
            for choice in choices
        ]
        q_input_ids = tokenized_question['input_ids']
        q_attention_mask = tokenized_question['attention_mask']
        options_input_ids = [opt['input_ids'] for opt in tokenized_options]
        options_attention_mask = [opt['attention_mask'] for opt in tokenized_options]
        
        label = sorted_keys.index(example['answer'])
        
        q_input_ids_list.append(q_input_ids)
        q_attention_mask_list.append(q_attention_mask)
        opt_input_ids_list.append(options_input_ids)
        opt_attention_mask_list.append(options_attention_mask)
        labels_list.append(label)
    
    q_input_ids_tensor = torch.tensor(q_input_ids_list)              # (num_samples, seq_len_q)
    q_attention_mask_tensor = torch.tensor(q_attention_mask_list)      # (num_samples, seq_len_q)
    opt_input_ids_tensor = torch.tensor(opt_input_ids_list)            # (num_samples, num_choices, seq_len_opt)
    opt_attention_mask_tensor = torch.tensor(opt_attention_mask_list)  # (num_samples, num_choices, seq_len_opt)
    labels_tensor = torch.tensor(labels_list)                          # (num_samples)
    
    dataset = torch.utils.data.TensorDataset(q_input_ids_tensor,
                                             q_attention_mask_tensor,
                                             opt_input_ids_tensor,
                                             opt_attention_mask_tensor,
                                             labels_tensor)
    return dataset

########################################
# DataLoaders, Optimizer, Scheduler, Accuracy
########################################

def get_data_loaders(train_dataset, val_dataset, batch_size):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset),
        batch_size=batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        batch_size=1
    )
    return train_dataloader, val_dataloader

def get_optimizer(model, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer

def get_scheduler(dataloader, optimizer, epochs):
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return scheduler

def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

########################################
# Training Loop
########################################

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data, val_data, num_choices = data_prep(args)
    
    # Load tokenizer from the encoder model.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets.
    train_dataset = create_dataset(train_data, tokenizer, num_choices)
    val_dataset = create_dataset(val_data, tokenizer, num_choices)
    
    # Get hidden size from the encoder config.
    config = AutoConfig.from_pretrained(args.model)
    hidden_size = config.hidden_size
    
    # Initialize the custom model with cross-attention.
    model = CustomMultipleChoiceCrossAttentionModel(encoder_name=args.model,
                                                      hidden_size=hidden_size).to(device)
    
    epochs = wandb.config.epochs
    train_dataloader, validation_dataloader = get_data_loaders(train_dataset, val_dataset, wandb.config.batch_size)
    
    optimizer = get_optimizer(model, wandb.config.learning_rate)
    scheduler = get_scheduler(train_dataloader, optimizer, wandb.config.epochs)
    
    total_t0 = time.time()
    best_accuracy = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch_i in range(epochs):
        print("")
        print(f'======== Epoch {epoch_i+1} / {epochs} ========')
        print('Training...')
        
        total_train_loss = 0
        total_train_accuracy = 0
        t0 = time.time()
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            q_input_ids = batch[0].to(device)
            q_attention_mask = batch[1].to(device)
            opt_input_ids = batch[2].to(device)
            opt_attention_mask = batch[3].to(device)
            b_labels = batch[4].to(device)
            
            model.zero_grad()
            logits = model(q_input_ids, q_attention_mask, opt_input_ids, opt_attention_mask)
            loss = loss_fn(logits, b_labels)
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            logits_cpu = logits.detach().cpu().numpy()
            labels_cpu = b_labels.cpu().numpy()
            total_train_accuracy += accuracy(logits_cpu, labels_cpu)
            
            if step % 5 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}.')
                avg_train_loss = total_train_loss / step
                wandb.log({'train_loss': avg_train_loss})
                
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print(f"  Training Accuracy: {avg_train_accuracy:.2f}")
        print(f"  Average Training Loss: {avg_train_loss:.3f}")
        print(f"  Training Epoch took: {format_time(time.time()-t0)}")
        
        wandb.log({'avg_train_accuracy': avg_train_accuracy,
                   'avg_train_loss': avg_train_loss,
                   'epoch': epoch_i})
        
        # Evaluation
        print("")
        print("Running evaluation...")
        t0 = time.time()
        model.eval()
        
        total_test_accuracy = 0
        total_test_loss = 0
        for batch in validation_dataloader:
            q_input_ids = batch[0].to(device)
            q_attention_mask = batch[1].to(device)
            opt_input_ids = batch[2].to(device)
            opt_attention_mask = batch[3].to(device)
            b_labels = batch[4].to(device)
            
            with torch.no_grad():
                logits = model(q_input_ids, q_attention_mask, opt_input_ids, opt_attention_mask)
                loss = loss_fn(logits, b_labels)
                total_test_loss += loss.item()
            
            logits_cpu = logits.detach().cpu().numpy()
            labels_cpu = b_labels.cpu().numpy()
            total_test_accuracy += accuracy(logits_cpu, labels_cpu)
        
        avg_test_accuracy = total_test_accuracy / len(validation_dataloader)
        avg_test_loss = total_test_loss / len(validation_dataloader)
        print(f"  Evaluation Accuracy: {avg_test_accuracy:.4f}")
        print(f"  Evaluation Loss: {avg_test_loss:.3f}")
        print(f"  Evaluation took: {format_time(time.time()-t0)}")
        
        wandb.log({'avg_test_accuracy': avg_test_accuracy, 'epoch': epoch_i})
        
        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            save_path = os.path.join(args.save_path, f"best_model_epoch_{epoch_i+1}.pth")
            # torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy {best_accuracy:.4f}")
        
        save_path = os.path.join(args.save_path, f"model_epoch_{epoch_i+1}.pth")
        # torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path} with accuracy {avg_test_accuracy:.4f}")
    
    print("Training complete!")
    print(f"Total training took: {format_time(time.time()-total_t0)}")
    print("best accuracy: ", best_accuracy)

########################################
# Main
########################################

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    # Initialize wandb.
    name = os.path.split(args.save_path)[-1] if args.save_path != "./" else None
    wandb.login()
    wandb.init(project="1508", entity="distill-llms", name=name,
               config={"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.bs})
    
    train(args)
