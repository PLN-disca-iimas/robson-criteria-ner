from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import classification_report as skl_classification_report
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from sklearn.metrics import confusion_matrix
from transformers import TrainingArguments
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers import Trainer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sklearn_crfsuite
import seaborn as sns
import numpy as np
import evaluate
import logging
import pickle
import torch
import time
import gc
import os

class RobsonCriteriaNERModel:
    def __init__(self) -> None:
        """
        Initialize the ModelInference class.
        """
        self.logger = logging.getLogger("RobsonCriteriaNERModel")
        self.logger.setLevel(logging.INFO)
        self.model_output_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = evaluate.load("seqeval")
        self.evaluations = []
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.dataset_path = None
        self.label_list = None
        self.short_model_name = None
        self.base_path = None
        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists("imgs"):
            os.makedirs("imgs")
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def define_logger(self, log_filename: str) -> None:
        """
        Defines the logger for the class.
        This method sets up the logger to log messages to a file and the console.
        Returns:
            None
        """
        log_folder = f"{self.base_path}/logs"
        os.makedirs(log_folder, exist_ok=True)
        log_filepath = os.path.join(log_folder, log_filename)
        handler = logging.FileHandler(log_filepath)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(f"Logger initialized with log file: {log_filename}")

    def set_base_path(self, base_path: str) -> None:
        """
        Sets the base path for loading/storing pre-trained models and log files.
        Args:
            base_path (str): The base path where the model and log files will be stored.
        Returns:
            None
        """
        self.base_path = base_path

    def load_model_and_tokenizer(self, model_name: str, short_model_name: str) -> None:
        """
        Loads the model and tokenizer from the specified pre-trained model name or local path.
        This method initializes the model for token classification and the tokenizer for processing text data.
        Args:
            model_name (str): The name of the pre-trained model or a local path to use.
            short_model_name (str): A short name for the model to be used in logging.
        Returns:
            None
        """
        self.short_model_name = short_model_name
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name, num_labels=len(self.label_list)
            ).to(self.device)
            self.logger.info(f"Model loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise e
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"Tokenizer loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading tokenizer {model_name}: {e}")
            raise e

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads the dataset from the specified path.
        Args:
            dataset_path (str): The path to the dataset to be loaded.
        Returns:
            None
        """
        try:
            self.dataset_path = dataset_path
            self.dataset = load_from_disk(dataset_path)
            self.label_list = self.dataset["train"].features["labels"].feature.names
        except Exception as e:
            self.logger.error(f"Error loading dataset from {dataset_path}: {e}")
            raise e

    def compute_elapsed_time(self, seconds: float) -> str:
        """Computes elapsed time in hours, minutes and seconds.
        Args:
            seconds (float): Elapsed time in seconds.
        Returns:
            A string with the elapsed time in the format HH:MM:SS.
        """

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"

    def flush(self) -> None:
        """
        Flush the memory to free up resources.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def tokenize_and_align_labels(self, examples: dict) -> dict:
        """
        Tokenizes the input examples and aligns the labels with the tokenized inputs.
        Args:
            examples (dict): A dictionary containing the input examples with "tokens" and "labels".
        Returns:
            dict: A dictionary containing the tokenized inputs and aligned labels.
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p: tuple) -> dict:
        """
        Computes the evaluation metrics for the model predictions.
        Args:
            p (tuple): A tuple containing the predictions and labels.
        Returns:
            dict: A dictionary containing the precision, recall, f1 score, and accuracy.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_labels = [
            [self.label_list[l] for l, p in zip(label, pred) if l != -100]
            for label, pred in zip(labels, predictions)
        ]
        true_predictions = [
            [self.label_list[p] for l, p in zip(label, pred) if l != -100]
            for label, pred in zip(labels, predictions)
        ]

        self.logger.info("\t\t\tSEQEVAL METRICS")
        self.logger.info(f'\t{"-" * 150}')
        seq_metric = seq_classification_report(
            y_true=true_labels,
            y_pred=true_predictions,
            zero_division=0.0,
            scheme="IOB2",
        )
        self.logger.info(f"\n{seq_metric}")
        self.logger.info(f'\t{"-" * 150}')

        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_true_predictions = [
            label for sublist in true_predictions for label in sublist
        ]
        tmp_labels = list(set(flat_true_labels))
        tmp_labels.remove("O")
        tmp_labels = sorted(
            tmp_labels, key=lambda k: (k.split("-")[1], k.split("-")[0])
        )
        self.logger.info("\t\t\tSKLEARN METRICS")
        self.logger.info(f'\t{"-" * 150}')
        skl_metric = skl_classification_report(
            y_true=flat_true_labels,
            y_pred=flat_true_predictions,
            labels=tmp_labels,
            zero_division=0.0,
        )
        self.logger.info(f"\n{skl_metric}")
        self.logger.info(f'\t{"-" * 150}')

        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        # self.evaluations.append(all_metrics)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def main(self, epochs: int, batch_size: int) -> None:
        """
        Main method to run the training and evaluation of the model.
        This method tokenizes the dataset, sets up the training arguments, and trains the model using the Trainer API.
        It also evaluates the model on the validation and test sets, logging the results.
        Args:
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size to use during training.
        Returns:
            None
        """
        os.makedirs(f"{self.base_path}/models", exist_ok=True)
        self.model_output_name = (
            f"{self.base_path}/models/snt_{self.short_model_name}_{epochs}_{batch_size}"
        )
        self.logger.info(f"Dataset loaded from {self.dataset_path}")
        self.logger.info(f"\n{self.dataset}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"CUDA version: {torch.version.cuda}")
        self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        self.logger.info(f"GPU model: {torch.cuda.get_device_name()}")
        self.logger.info(f"Model output name: {self.model_output_name}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch size: {batch_size}\n")
        self.logger.info("Training the model...\n")
        start_time = time.time()
        tokenized_datasets = self.dataset.map(
            self.tokenize_and_align_labels, batched=True
        )

        columns_to_remove = ["snt_id", "note_id", "pos"]
        tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_output_name,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            no_cuda=False,  # Ensure this is set to False to use CUDA
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            data_collator=data_collator,
            # tokenizer=tokenizer, is deprecated and will be removed in version 5.0.0
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.logger.info(
            f"Training completed in {self.compute_elapsed_time(trainer.state.log_history[-1]['train_runtime'])}"
        )
        self.logger.info(f"\tTraining completed.")
        self.logger.info(f"\tEvaluating the model on the test set...\n")
        metrics = trainer.evaluate(tokenized_datasets["test"])
        self.logger.info(f"\t{metrics}")
        # get the best model path
        self.logger.info(f"\tBest model path: {trainer.state.best_model_checkpoint}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = self.compute_elapsed_time(elapsed_time)

        logging.info(f"\tTime elapsed:\t{formatted_time}")
        self.logger.info("\tDone.\n")

        self.flush()
        del self.model
        del self.tokenizer

    def error_analysis(self, predictions: list) -> None:
        """
        Perform error analysis on the model predictions and labels.
        Args:
            predictions (list): The model predictions.
            labels (list): The true labels.
        Returns:
            None
        """
        self.logger.info("Performing error analysis...")
        # Implement error analysis logic here
        # This could include analyzing misclassifications, etc.
        self.logger.info("Error analysis completed.")


class RobsonCriteriaCRFNERModel:

    def __init__(
        self,
        base_path: str,
        iterations: int,
        model_output_name: str,
    ) -> None:
        """
        Initialize the ModelInference class.
        """
        self.logger = logging.getLogger("RobsonCriteriaCRFNERModel")
        self.logger.setLevel(logging.INFO)
        self.base_path = base_path
        self.iterations = iterations
        self.model_output_name = model_output_name
        self.dataset = None
        self.label_list = None
        self.pos_list = None
        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists("imgs"):
            os.makedirs("imgs")
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def define_logger(self, log_filename: str) -> None:
        """
        Defines the logger for the class.
        This method sets up the logger to log messages to a file and the console.
        Returns:
            None
        """
        log_folder = f"{self.base_path}/logs"
        os.makedirs(log_folder, exist_ok=True)
        log_filepath = os.path.join(log_folder, log_filename)
        handler = logging.FileHandler(log_filepath)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(f"Logger initialized with log file: 'logs/{log_filename}'")

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads the dataset from the specified path.
        Args:
            dataset_path (str): The path to the dataset to be loaded.
        Returns:
            None
        """
        try:
            self.dataset_path = dataset_path
            self.dataset = load_from_disk(dataset_path)
            self.pos_list = self.dataset["train"].features["pos"].feature.names
            label_list = self.dataset["train"].features["labels"].feature.names

            self.logger.info(f"\t{dataset_path=}")
            self.logger.info(f"\n{self.dataset}\n")

            # replace accented letters
            for i, label in enumerate(label_list):
                label = (
                    label.replace("á", "a")
                    .replace("é", "e")
                    .replace("í", "i")
                    .replace("ó", "o")
                    .replace("ú", "u")
                )
                label_list[i] = label
            self.label_list = label_list

        except Exception as e:
            self.logger.error(f"Error loading dataset from {dataset_path}: {e}")
            raise e

    def compute_elapsed_time(self, seconds: float) -> str:
        """Computes elapsed time in hours, minutes and seconds.
        Args:
            seconds (float): Elapsed time in seconds.
        Returns:
            A string with the elapsed time in the format HH:MM:SS.
        """

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"

    def build_data_to_conll(self, rows, labels, pos_tags, set_name):
        data = []
        for row in tqdm(
            rows, total=len(rows), desc=f"Transforming {set_name} to CoNLL format"
        ):
            data.append(
                [
                    (tkn, pos_tags[pos], labels[lbl])
                    for (tkn, pos, lbl) in zip(row["tokens"], row["pos"], row["labels"])
                ]
            )

        logging.info(f"\t{len(rows):,} sentences in the '{set_name}' set")

        return data

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "postag": postag,
            "postag[:2]": postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:word.istitle()": word1.istitle(),
                    "-1:word.isupper()": word1.isupper(),
                    "-1:postag": postag1,
                    "-1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:word.istitle()": word1.istitle(),
                    "+1:word.isupper()": word1.isupper(),
                    "+1:postag": postag1,
                    "+1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["EOS"] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, postag, label in sent]

    def remove_bio_prefix(self, labels):
        """
        Remove the BIO scheme from the tags
        """
        return [label[2:] if label != "O" else "O" for label in labels]

    def get_confusion_matrix(
        self, y_true: list[str], y_pred: list[str], labels: list[str], fig_name: str
    ) -> None:
        """
        Plot a confusion matrix

        Args:

        Returns:
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(20, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Gold")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"imgs/{fig_name}.png", dpi=300, bbox_inches="tight")
        self.logger.info(f"\tFigure saved in 'imgs/{fig_name}.png' ")

    def tranform_dataset(self, data_name: str) -> list:
        return self.build_data_to_conll(
            self.dataset[data_name],
            self.label_list,
            self.pos_list,
            data_name,
        )

    def main(self):
        start_time = time.time()
        self.logger.info(f"Transforming data to CoNLL format")
        train_data = self.tranform_dataset("train")
        test_data = self.tranform_dataset("test")

        # extract features from the data
        self.logger.info(f"\tExtracting features from the data")
        X_train = [self.sent2features(s) for s in train_data]
        X_test = [self.sent2features(s) for s in test_data]

        y_train = [self.sent2labels(s) for s in train_data]
        y_test = [self.sent2labels(s) for s in test_data]

        self.logger.info(f"\t\tX_train = {len(X_train):,}")
        self.logger.info(f"\t\ty_train = {len(y_train):,}")
        self.logger.info(f"\t\tX_test = {len(X_test):,}")
        self.logger.info(f"\t\ty_test = {len(y_test):,}")

        self.logger.info(f"\tTraining the model")
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=self.iterations,
            all_possible_transitions=True,
            verbose=False,
        )

        self.logger.info(f"\tFitting the model...")
        crf.fit(X_train, y_train)

        # evaluate test dataset
        self.logger.info(f"\tEvaluating on the test set...")
        y_test_predict = crf.predict(X_test)

        # prepare y_test labels
        labels = list(set([tag for sublist in y_test for tag in sublist]))
        labels.remove("O")
        labels = sorted(
            labels,
            key=lambda k: (
                k.split("-")[1],
                k.split("-")[0],
            ),
        )

        # remove BIO prefix
        y_true_simple = self.remove_bio_prefix(
            [tag for sublist in y_test for tag in sublist]
        )
        y_pred_simple = self.remove_bio_prefix(
            [tag for sublist in y_test_predict for tag in sublist]
        )

        # prepare labels to confusion matrix
        cm_labels = list(set(y_true_simple))
        cm_labels.remove("O")
        cm_labels.sort()
        # confusion matrix witouth BIO schema
        self.get_confusion_matrix(
            y_true_simple, y_pred_simple, cm_labels + ["O"], f"crf_cmd_{self.iterations}"
        )

        # classification report
        seq_metric = seq_classification_report(
            y_true=y_test,
            y_pred=y_test_predict,
            zero_division=0.0,
            scheme="IOB2",
        )

        self.logger.info(f"{'-'*80}\n\t\tSeqeval metrics\n{'-'*80}")
        self.logger.info(f"\n{seq_metric}\n")

        y_true = [tag for sublist in y_test for tag in sublist]
        y_pred = [tag for sublist in y_test_predict for tag in sublist]

        skl_metric = skl_classification_report(
            y_true=y_true, y_pred=y_pred, labels=labels, zero_division=0.0
        )
        self.logger.info(f"{'-'*80}\n\t\tSklearn metrics\n{'-'*80}")
        self.logger.info(f"\n{skl_metric}\n")

        # confusion matrix with BIO schema
        self.get_confusion_matrix(
            y_true, y_pred, labels + ["O"], f"crf_cmd_{self.iterations}_BIO"
        )

        # save the crf model as pickle file
        model_name = ''
        if self.model_output_name:
            model_name = f"model/{self.model_output_name}.pkl"
        else:
            model_name = f"model/crf_{self.iterations}_snts.pkl"

        with open(model_name, "wb") as f:
            pickle.dump(crf, f)

        self.logger.info(f"\tModel saved to '{model_name}'")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = self.compute_elapsed_time(elapsed_time)

        self.logger.info(f"\tTime elapsed:\t{formatted_time}\n")
