import time
import torch.utils
import torch.utils.data
import pandas as pd
import numpy as np
import argparse
import os
import datetime
import wandb
import json
import nlpaug.augmenter.word as naw
import random
import copy

import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice



def data_prep(args):
    # Load the JSON files directly into lists of dictionaries
    with open("train.json", "r") as f:
        train_data = json.load(f)
    with open("test.json", "r") as f:
        val_data = json.load(f)

    # Apply augmentation if specified
    if args.aug:
        # Augment with synonyms using WordNet
        # Initialize augmenters
        aug_syn = naw.SynonymAug(aug_src='wordnet')
        aug_spell = naw.SpellingAug()

        new_train_samples = []
        for sample in train_data:
            # Create a deep copy of the sample for each augmentation type.
            sample_syn = copy.deepcopy(sample)
            sample_spell = copy.deepcopy(sample)
            
            # Augment using synonym augmentation
            sample_syn['query'] = aug_syn.augment(sample_syn['query'])[0]
            for key in sample_syn['options'].keys():
                sample_syn['options'][key] = aug_syn.augment(sample_syn['options'][key])[0]
            
            # Augment using spelling augmentation
            sample_spell['query'] = aug_spell.augment(sample_spell['query'])[0]
            for key in sample_spell['options'].keys():
                sample_spell['options'][key] = aug_spell.augment(sample_spell['options'][key])[0]
            
            # Append the augmented copies to the list of new samples.
            new_train_samples.append(sample_syn)
            # new_train_samples.append(sample_spell)
            
        # Concatenate original train data with new augmented samples.
        train_data = train_data + new_train_samples

    if args.debug:
        train_data = train_data[:10]
        val_data = val_data[:10]

    # Determine the number of choices; assuming each sample's 'options' field is a dict.
    num_choices = len(train_data[0]['options'])

    return train_data, val_data, num_choices


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save-path',
        type=str,
        required=False,
        default="./",
        help='The file path where output file will be saved.'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='roberta-base',
        help='The name of the model to be used.'
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        required=False,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        required=False,
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        required=False,
    )
    parser.add_argument(
        '--aug',
        action='store_true',
        default=False,
        help='apply data augmentation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug'
    )

    args = parser.parse_args()
    return args


def load_model(args, model_name, tokenizer_name, num_choices, output_attentions=False, output_hidden_states=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
    # model.config.hidden_dropout_prob = 0.2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return model, tokenizer


def create_dataset(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for example in examples:
        query = example['query']
        options = example['options']
        # Use sorted keys
        sorted_keys = sorted(options.keys())
        choices = [options[key] for key in sorted_keys]

        # Tokenize each (query, option) pair
        tokenized_choices = [
            tokenizer(query, choice, add_special_tokens=True, truncation=True, max_length=128, padding='max_length')
            for choice in choices
        ]
        input_ids = [tc['input_ids'] for tc in tokenized_choices]
        attention_mask = [tc['attention_mask'] for tc in tokenized_choices]

        # Determine the label index based on the key matching the answer.
        label = sorted_keys.index(example['answer'])
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)

    input_ids_tensor = torch.tensor(input_ids_list) # [400, 5, 128] num_samples, options per sample, context length
    attention_mask_tensor = torch.tensor(attention_mask_list) # [400]
    labels_tensor = torch.tensor(labels_list)

    dataset = torch.utils.data.TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
    return dataset


def get_data_loaders(train_dataset, val_dataset, generator):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset),
        batch_size=wandb.config.batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        batch_size=1
    )

    return train_dataloader, val_dataloader


def get_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=wandb.config.learning_rate, 
                      eps=1e-8)
    return optimizer


def get_scheduler(dataloader, optimizer):
    total_steps = len(dataloader) * wandb.config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return scheduler


def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(args):

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # generator = torch.Generator().manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For multiple choice, data_prep returns a list of dict examples.
    train_data, val_data, num_choices = data_prep(args)

    model, tokenizer = load_model(args, args.model, args.model, num_choices,
                                  output_attentions=False, output_hidden_states=False)

    # Create datasets from the examples
    train_dataset = create_dataset(train_data, tokenizer)
    val_dataset = create_dataset(val_data, tokenizer)

    epochs = wandb.config.epochs

    # Get data loaders
    train_dataloader, validation_dataloader = get_data_loaders(train_dataset, val_dataset, generator=None)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(train_dataloader, optimizer)

    total_t0 = time.time()
    best_accuracy = 0
    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_accuracy = 0
        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            # For multiple choice, b_input_ids shape is (batch_size, num_choices, seq_len)
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels).to_tuple()

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 5 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                avg_train_loss = total_train_loss / step
                wandb.log({'train_loss': avg_train_loss})

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            total_train_accuracy += accuracy(logits, label_ids)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("  accuracy: {0:.2f}".format(avg_train_accuracy))
        training_time = format_time(time.time() - t0)

        wandb.log({'avg_train_accuracy': avg_train_accuracy,
                   'avg_train_loss': avg_train_loss,
                   'epochs': epoch_i,})

        print("")
        print("  average training loss: {0:.3f}".format(avg_train_loss))
        print("  training epoch took: {:}".format(training_time))

        print("")
        print("Running testing...")

        t0 = time.time()
        model.eval()

        total_test_accuracy = 0
        total_test_loss = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                val_loss, val_logits = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels).to_tuple()

            val_logits = val_logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            total_test_accuracy += accuracy(val_logits, label_ids)

        avg_test_accuracy = total_test_accuracy / len(validation_dataloader)
        print("  accuracy: {0:.4f}".format(avg_test_accuracy))

        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            save_path = os.path.join(args.save_path, f"best_model_epoch_{epoch_i + 1}.pth")
            # torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy {best_accuracy:.4f}")
        save_path = os.path.join(args.save_path, f"model_epoch_{epoch_i + 1}.pth")
        # torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path} with accuracy {avg_test_accuracy:.4f}")

        testing_time = format_time(time.time() - t0)
        wandb.log({'avg_test_accuracy': avg_test_accuracy, 'epochs': epoch_i,})
        print("  testing took: {:}".format(testing_time))

    print("training complete!")
    print("total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    print("best accuracy: ", best_accuracy)


if __name__ == "__main__":

    

    args = parse_arguments()
    print(args)

    # Initialize wandb as needed; for example:
    name = os.path.split(args.save_path)[-1] if args.save_path != "./" else None
    wandb.login()
    wandb.init(project="1508", entity="distill-llms", name=name, config={"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.bs})
    train(args)
