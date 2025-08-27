from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from datasets import load_from_disk
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm.auto import tqdm
from trl import SFTTrainer
import logging
import torch


def get_ettities_by_note(note_id, tokens, labels):
    entities = {}
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            entity = labels[i][2:]
            tmp = tokens[i]
            j = i + 1
            while j < len(labels) and labels[j] == "I-" + entity:
                tmp += " " + tokens[j]
                j += 1
            if entity not in entities:
                entities[entity] = [tmp]
            else:
                entities[entity].append(tmp)
            i = j
        elif labels[i].startswith("I-"):
            print(f"\tError: the entity {labels[i]} is not starting with 'B-'")
            print(f"\t{note_id=}")
            print(f"\t{i=}")
            print(f"\t{labels[i]=}")
            exit(0)
        elif labels[i] == "O":
            i += 1
        else:
            print("\tError: something was wrong")
            print(f"\t{note_id=}")
            print(f"\t{i=}")
            print(f"\t{labels[i]=}")
            exit(0)
    return entities


def prepare_data(data, tags, instruction, name) -> dict[str, any]:
    # train = aligning_labels(data['train'], labels, instruction)
    instructions = []
    context = []
    response = []
    for note in tqdm(data, total=len(data), desc=f"Preparing {name}"):
        labels = [tags[label] for label in note["tags"]] # gold_notes_80-20
        tokens = note["tokens"]
        entities = get_ettities_by_note(note["id"], tokens, labels)

        instructions.append(instruction)
        context.append(" ".join(tokens))
        response.append(entities)

    return {
        "instruction": instructions,
        "context": context,
        "response": response,
    }


if __name__ == "__main__":
    log_path = f"logs/llama_gold_finetuning_3_epochs.log" # gold_notes_80-20

    # reset file
    g = open(log_path, "w")
    g.close()

    # config logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s\t%(message)s",
    )

    # load the dataset
    data_path = "data/gold_notes_80-20"
    data = load_from_disk(data_path)

    # tags to convert their ids to labels
    tags = data["train"].features["tags"].feature.names # gold_notes_80-20

    # instruction defined to do the prompting
    instruction = "You are an expert identifying clinical entities on Spanish notes in the obstetric domain, entities are related to the obstetric area, as well as antibiotics, uterotonics, their doses, and posology, and those entities with personal information about the patient, their family members, and medical staff."

    # prepare the dataset: {instruction, context, response}
    train_dataset = Dataset.from_dict(
        prepare_data(data["train"], tags, instruction, "train")
    )
    eval_dataset = Dataset.from_dict(
        prepare_data(data["validation"], tags, instruction, "valid") # gold_notes_80-20
    )

    # download the 4-bit model
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # format the text columns using the Alpaca format
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n

    ### Instruction:\n
    {}\n\n

    ### Input:\n
    {}\n
    {}\n\n

    ### Response:\n
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["context"]
        outputs = examples["response"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, EOS_TOKEN, output)
            texts.append(text)
        return {
            "text": texts,
        }

    # map the formatting function on train and eval sets
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    # -------------------------------------------------------------------------------------
    # the LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # for rank 32
    params = model.print_trainable_parameters()
    logging.info(params)
    # trainable params: 65,011,712 || all params: 8,095,272,960 || trainable%: 0.8031

    # -------------------------------------------------------------------------------------
    # the Trainer & the Training

    class CustomCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])

    callback = CustomCallback()

    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # how many samples to train at a time per step per device
        gradient_accumulation_steps=4,  # total steps will be divided by this
        warmup_steps=2,  # steps to go from 0 to specified lr
        num_train_epochs=3,  # no of samples per step * this would be total steps
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="gold_outputs_3",
        eval_strategy="steps",  # Evaluate during training
        eval_steps=1,  # Evaluate every 1 step
        save_strategy="steps",
        save_steps=10,  # save model state every 10 steps
        load_best_model_at_end=True,  # Load the best model when finished training
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # pass in the training split
        eval_dataset=eval_dataset,  # pass in the eval split
        args=training_args,
        dataset_text_field="text",
        callbacks=[callback],  # our custom callback
    )

    # check GPU memory before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logging.info(f"{start_gpu_memory} GB of memory reserved.")

    # start training
    trainer_stats = trainer.train()

    # check GPU memory after training
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logging.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logging.info(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    logging.info(f"Peak reserved memory = {used_memory} GB.")
    logging.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logging.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logging.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    # -------------------------------------------------------------------------------------
    # Plot training loss for train & eval sets
    plt.figure(figsize=(12, 6))
    plt.plot(callback.train_losses, label="Train Loss")
    plt.plot(callback.eval_losses, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.savefig("imgs/gold_fine_tuning_3_epochs.png", dpi=300, bbox_inches="tight")
    # plt.show()

