# ASBA (or as everybody else says, ABSA)
## Aspect-based sentiment analysis

TO DO:
1) fix evaluation

Pipeline:
- fine-tune BERT, roBERTa and maybe DeBerta on ABSA task using SemEval-2014 Task-4 (https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2)
- you can find the used datasets in this repo under the folder [Datasets]
- Code for our deep learning assignment

We used huggingfaces pre-trained models to fine-tune for the task and their tokenizers to encode the input. Find below a summary of the necessary steps to finetune a BERT model to classify both the aspect and the polarity of a sentence (based on the aspect).

In order to output both the aspect and polarization of the sentence based on the aspect, we can frame this task as a sequence classification problem where the input sequence consists of the sentence concatenated with the aspect, and the model outputs both the aspect and its corresponding polarization.

Data Preparation:

prepare the dataset with sentences, aspects, and polarizations. Encode the sentences and aspects using BERT's tokenizer.
Concatenate the encoded sentence and aspect tokens. You may need to add special tokens (e.g., [SEP]) to separate them.

Model Architecture:

Modify the top layers of BERT for sequence classification to output both the aspect and polarization.
You can add additional layers on top of BERT's output to predict the aspect and polarization separately.
For instance, you can use two separate linear layers followed by softmax activations to predict aspect and polarization.

Loss Function:

Define a suitable loss function for both aspect and polarization predictions. This could be a combination of cross-entropy loss for aspect prediction and mean squared error (MSE) or cross-entropy loss for polarization prediction, depending on the nature of your polarization labels.
Training:

Fine-tune the modified BERT model on your dataset using the concatenated input sequences.
During training, minimize the combined loss of aspect and polarization predictions.
Iterate over the dataset for multiple epochs and update the model parameters using an optimizer.

Evaluation:

Evaluate the performance of the fine-tuned model on a validation set or through cross-validation.
Measure accuracy, precision, recall, F1-score, or any other relevant metrics for both aspect and polarization predictions.

