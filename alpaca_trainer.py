from alpaca_dataset import AlpacaData
from transformers import (
    TrainingArguments,
    Trainer,
)


class AlpacaTrainer:
    def __init__(self, model, tokenizer, train, val):
        self.model = model
        self.tokenizer = tokenizer
        self.train = train
        self.val = val

    def train(self, train_args):
        dataset = self.dataset
        split = dataset.split_train_val()

        train_dataset = AlpacaData(self.tokenizer, 512, split["train"])
        val_dataset = AlpacaData(self.tokenizer, 512, split["validation"])

        args = TrainingArguments(**train_args)
        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        trainer.save_model()
