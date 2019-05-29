# Objectives

The learning objectives of this project are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

# Problem Description

Given: A tweet

Task: classify the tweet as zero or more of eleven emotions that best represent the mental state of the tweeter:

1. anger (also includes annoyance and rage) can be inferred
2. anticipation (also includes interest and vigilance) can be inferred
3. disgust (also includes disinterest, dislike and loathing) can be inferred
4. fear (also includes apprehension, anxiety, concern, and terror) can be inferred
5. joy (also includes serenity and ecstasy) can be inferred
6. love (also includes affection) can be inferred
7. optimism (also includes hopefulness and confidence) can be inferred
8. pessimism (also includes cynicism and lack of confidence) can be inferred
9. sadness (also includes pensiveness and grief) can be inferred
10. suprise (also includes distraction and amazement) can be inferred
11. trust (also includes acceptance, liking, and admiration) can be inferred

# Evaluation Metrices

The primary evaluation metric is multi-label accuracy (or Jaccard index). Since this is a multi-label classification task, each tweet can have one or more gold emotion labels, and one or more predicted emotion labels. Multi-label accuracy is defined as the size of the intersection of the predicted and gold label sets divided by the size of their union. This measure is calculated for each tweet, and then is averaged over all tweets in the dataset.

Secondary evaluation metrics are also provided: micro-averaged F-score and macro-averaged F-score. These additional metrics are intended to provide a different perspective on the results.

# Results on test set

1. Accuracy - 51.2%
2. Micro-avg F1 - 62.7%
3. Macro-avg F1 - 46.8%
