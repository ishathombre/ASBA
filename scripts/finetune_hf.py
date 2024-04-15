# we define the metrics to be computed during training, the crossnetropy loss and accuracy.
from torch.nn import CrossEntropyLoss
# we define the training loop and arguments under a function, using the Trainer from huggigfaces
from transformers import TrainerCallback
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels)).item()
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"loss": loss, "accuracy": accuracy}

# Training function
def training_run(model, output_dir, num_epochs, batch_size, train_dataset, val_dataset):

    epochs = 3
    batch_size = 8
    num_steps = len(train_dataset) * epochs // batch_size
    warmup_steps = num_steps // 10  # 10% of the training steps
    save_steps = num_steps // epochs
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    #criterion = CrossEntropyLoss()


    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        warmup_steps = warmup_steps,
        weight_decay = 0.01,
        logging_dir = 'logs',
        logging_steps = 10,
        evaluation_strategy = 'steps',
        do_eval = True,
        learning_rate = 5e-5, #0.00005
        save_steps = save_steps)

    trainer = Trainer(model,
                      training_args,
                      #optimizers=optimizer,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      compute_metrics=compute_metrics)


    # Start training
    trainer.train()
    #trainer.evaluate()
