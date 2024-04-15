# ASBA (or as everybody else says, ABSA)
## Aspect-based sentiment analysis

Pipeline:
- we fine-tuned BERT, roBERTa and DeBerta on ABSA task using SemEval-2014 Task-4 (https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2) --> find the code used in the colab notebook, you can run the testing on the checkpoints provided - downoload available in the notebook - just keep in mind that it will may take a while for running the testing on CPU (approx. 30 minutes) --> the checkpoints for this task can be found here: \href{https://drive.google.com/drive/folders/19x741pTLTDx_SQC2Dm5MgHQskEu5_kp4?usp=sharing}
- we modified the models with a custom class to add 2 linear layers on top of them for double classification (aspects and polarities)
- you can find the used datasets in this repo under the folder [Datasets]

We used huggingfaces pre-trained models to fine-tune for the task and their tokenizers to prepare the input. For the second task we built on the pre-trained models from hf with a custom class to modify the architectures, using an "off-the-shelf" approach.
