# quora-duplicate-questions

                 QUORA duplicate questions classification problem







Introduction:
The problem of classifying duplicate questions or semantic similarity is one of the key research fields in the NLP field, as any NLP problem a precise pipeline is usually followed to get the final model that is best suited for the problem in question. The first step is text preprocessing in an attempt to normalize the input data, then data exploration is done to get a sense of the data distribution and structure, then follows the most important part which is features extraction and exploration, embeddings being the most promising features in the state of the art papers and competitions submissions, finally features selection and model training and evaluation. To make sure the same pathway was followed in the proposed solution, and due to the number of options and combinations I had to choose from during each step, I tried to keep the code modularized as much as possible to avoid code duplication and to simplify the trial and error phase while choosing the options with the strongest argument supporting them as a final solution. The code was run on a 25GB disk and 12GB memory Google Colab instance, to bypass the Hardware limitations and to make sure the results are reliable, the pipeline was run several times on a fraction of data while averaging the final results, despite that some options were discarded due to the inability of the available hardware to tackle them even on the limited data size.
The figure below (Fig-1) summarizes the pipeline that was implemented on the notebook solution and the options available for each step.
                        
                                                          Fig-1

Building the NLP pipeline :
Below is an exploration of every step of the pipeline build on the notebook solution:

Data set exploration:
The main thing to retain after the data exploration phase is the fact that the data is unbalanced in favor of the non-duplicate questions class (Fig-2), which is something I have taken into consideration while choosing the models, accuracy metrics, and preprocessing methods. Further exploration should be done on extracted features in a univariate and bivariate way during future iterations.

                                                                 Fig-2

Data Preprocessing and sampling :
To process data, I have chosen to remove stop words, normalize typos, and lemmatize the text, to normalize the text questions across the whole data set, further typos could be extracted in future iterations.
This step should be used with caution because some typos may have predictive power, therefore removing them or standardizing the text may make the final model’s accuracy worse, that’s why I have chosen to persist both the original and the clean text versions, to test out in future iterations if normalizing the questions improves or decreases the accuracy of the models.

Features Extraction :
There are three types of features that could be extracted from the input data: 

NLP features: These features are calculated based on words or sentence ratios and are divided into two categories normal NLP features representing ratios extracted from both sentences or from their combination such as the questions lengths, and fuzzy ratios extracted based on the fuzzy text matching method such as the fuzzy and fuzzy partials ratios.

Graphical features: These features are extracted from the graph that represents the questions as nodes and their neighbors as edges, then measures are calculated for each node such as the Pager Rank, the problem with this approach is that most questions have only one neighbor and therefore graphical metrics are null or constant in most rows.

I have chosen to work on the original data while extracting NLP and Graphical features because these features are morphological in nature, and don't take into consideration the questions semantic or meaning which is what data preprocessing tries to normalize at the end. On the other hand, I have used the cleaned version while extracting embeddings, due to the same reason highlighted above, but to support this decision I would test out both versions to check for their impact on the final model performance.


Embeddings: There are many ways to create embeddings and embeddings extracted features.
Word2vec and Glove: These embeddings are calculated for each word, therefore to create an embedding for the whole questions sentences, I had two options, either calculate the sum or the TFIDF-Count weighted sum of the word2vec embeddings, the first option was implemented on the solution notebook, the second one was implemented but discarded until future iterations due to memory constraints.
Fasttext : The main advantage of Fasttext is its ability to handle out of vocab embeddings, and therefore it can be used on un-cleaned data because each word can be embedded, but due to the size of the Fasttext embedding matrices and the memory constraints, this option was implemented but discarded until further iterations.
BERT : Bert is a well-documented transformer model, with a strong record in language modeling, the input’s embedding is represented by the model’s last hidden layer, which can be used as an input to the features set.
For this architecture, I have chosen a pre-trained BERT model trained on the QUORA duplicate questions dataset.
XLNET : XLNET is the strongest candidate because it beats other languages in the QUORA duplicate questions benchmark according to the papers with code website https://paperswithcode.com/sota/question-answering-on-quora-question-pairs, but unfortunately training such models requires TPUs and larger memory, therefore to test out the ability of XLNET to classify questions I have chosen a pre-trained version.
The original embedding vectors or their compressed version extracted with dimension reduction algorithms such as SVD for example could be added to the features set, but due to hardware restrictions and due to the fact that similar questions should be as close as possible in term of distance and therefore what matters the most is the distance between those vectors and not their actual values, I have decided to calculate several distances metrics for each embedding option instead of the actual embedding values.
For each model, a suitable framework has been chosen, taking into account it’s simplicity or suitability for the task, other frameworks such as Embedding as a service and other embedding methods such as Google universal sentence encoder could be used in future iterations.
To estimate the number of memory and computational time needed to run the pipeline on the whole data, I have created an initialization method that initializes and tests the pipeline code and returns the estimated memory and calculation time needed to run the NLP pipeline, while working on a 0.001% fraction of data and keeping the proportion of duplicate and non-duplicate questions constant during sampling to avoid selection bias.  

Features selection:

Due to a large number of features, the next step was extracting the most important ones, by using model-based extraction methods and statistical methods such as the mutual information score to get the features with the highest variance, in order to compensate for
the sample size, I have decided to run this method several times while accumulating the importance of the features during each step. 
In general, reducing the number of features is a rule of thumb when dealing with a large set of features but to support this decision I have decided to train the models on both versions.


Training the Models:
I have decided to train XGBOOST and Light GBM models on the selected features because they perform well on similar problems especially in the case of unbalanced data sets.
In a future iteration, I would add more models and stack them while fine-tuning the hyperparameters for each model.
Other models such as the SIAMESE neural networks show superior results for this kind of problem https://www.aclweb.org/anthology/R19-1116/ , but they involve another preprocessing approach because they take as input the embedding of each question and classify their similarity based on those vectors.


Results and Interpretations:
Running the features selection method, shows clearly that Bert embedding distances are the strongest features according to all metrics, which is kind of obvious because the model was trained on the QUORA duplicate questions data set (Fig-3).


                                                                    Fig-3


I have decided to discard questions ID hashes despite being the most predictive features because I didn’t see any explanation of their high performance, and I have decided to keep the final models explainable to a certain level.
Training both models on the selected features gave acceptable results for a production environment in terms of accuracy and F1 scores (Fig-4),F1 score being the chosen comparaison accuracy metric due to the unbalanced nature of the data set, fine-tuning the hyperparameters, and stacking the models should in theory improve the accuracy even further.




                                                              Fig-4

Using SMOTE and training the models on the whole features sets, decreases the models’ performance which proves that the chosen approach is the optimal one (Fig-5).

                                                              Fig-5  

Clustering :
Clustering the questions involves another approach, each question should be handled individually, the features involved in the clustering process are one question centric in nature, and the difference between different questions doesn’t matter anymore, what matters in this case, is each question embedding,  therefore I have decided to create a data frame with stacked questions and their BERT embedding because it was the most effective embedding model for the classification problem, therefore I can assume with confidence that this model is a representative language model for this dataset, then I have decided to apply the BIRCH clustering algorithm to cluster the data, BIRCH is a fast clustering algorithm that performs well when there is a high number of features, other clustering algorithms such as DBSCAN or OPTICS could be used in a future iteration,
To explore the clusters I have decided to create a Word Cloud for each cluster to get a sense of the clusters’ topic, this method is also useful for finding the optimal number of clusters, increasing the number of clusters until each cluster’s subject is coherent and compact is a method I have used to find the optimal cluster number.
As an example, the figure below (Fig-6) shows clearly that cluster 1 contains questions relative to health and a healthy lifestyle.


                                                                     Fig-6

A topic modeling pre-trained model could be used in future iterations.

Data cleanliness and modularization :
Personal care was taken for code cleanness and modularization, by keeping the code aligned with the Google style guide, and the OOP SOLID principles especially the single responsibility principle and the open-closed principle, to make sure that the code is modularized and open for future modifications.

References:
https://github.com/Wrosinski/Kaggle-Quora
https://github.com/aerdem4/kaggle-quora-dup
https://github.com/stys/kaggle-quora-question-pairs
https://github.com/dysdsyd/kaggle-question-pairs-quora

