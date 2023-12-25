# Hate Speech Detection
This project is for the final project of Natural Language Processing course. The goal of this project is to show that the models trained on combined dataset can give us better generalization. 

Special thanks to my teammates Ethan Oh and Fernando Vera Buschmann for their valuable contribution to this project. 

we selected three datasets: AHSD, MHS and HATEX.
For our experiment, we implement two models: BERT and BERTweet.

We use the pretrained BERT and BERTweet model from Hugging Face Transformers library and fine tune them on three different datasets: AHSD, MHS and HATEX individually. Then in separate experiments, we fine tune these two models on the combined dataset consisting of these three datasets. For this project, we experiment for both binary classification and multiclass classification. Therefore, for binary classification, we fine tune four models: three for individual datasets and one for the combined dataset. Furthermore, for multiclass classification, we fine tune four more models. Here, our goal is to establish a statement that each model fine tuned on the combined dataset provides better evaluation scores on the individual datasets compared to the models fine tuned on the individual datasets. We used one GPU for the experiment and it took around 120 hours to fine tune these eight models. The batch size we use in our experiment is 32 and we ran all the experiments for 50 epochs. 

From each experiment, we saved the best models while training for 50 epochs and used them to evaluate on the test datasets. We assess all the fine tuned BERT and BERTweet models using relevant evaluation metrics including precision, recall, F1 score, and accuracy.

![binary](https://github.com/alshahriarrubel/HateSpeechDetect/assets/24860187/ba5855b4-5d66-4086-adf5-723b3c098ab2)

![multi](https://github.com/alshahriarrubel/HateSpeechDetect/assets/24860187/cd89e849-44ab-4843-b729-afa7f681e9d9)



## How to Run
* run .py files of BERT and BERTweet for both binary and multiclass with 4 datasets: AHSD, MHS, HATEX and Merged datasets. It will save the best models.
* run 4 .py files on evaluatation of BERT and BERTweet for both binary and multiclass. 

You will find more details about this work [here](https://github.com/onahte/hateDatasetStudy) when it will be publicly available.
