# prove-youtube
Prediction of views, likes, dislikes of youtube trending videos.

## Data
https://www.kaggle.com/datasnaek/youtube-new/downloads/youtube-new.zip/55

## Methods
1. Prediction is based on category id of video.
2. TF-IDF of text attributes of video (title, channel title, description, tags).
3. [Paragraph2Vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) (tensorflow implementation).
4. [SIF](https://openreview.net/pdf?id=SyK00v5xx).
