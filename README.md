# ASBA (or as everybody else says, ABSA)
## Aspect-based sentiment analysis

Pipeline:
- we fine-tuned BERT, roBERTa and DeBerta on ABSA task using SemEval-2014 Task-4 (https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2)
- we modified the models with a custom class to add 2 linear layers on top of them for double classification (aspects and polarities)
- you can find the used datasets in this repo under the folder [Datasets]

We used huggingfaces pre-trained models to fine-tune for the task and their tokenizers to prepare the input. For the second task we built on the pre-trained models from hf with a custom class to modify the architectures, using an "off-the-shelf" approach.
