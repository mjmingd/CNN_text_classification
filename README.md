# CNN_text_classification

This is implementation of the paper [Do Convolutional Networks need to be Deep for Text Classification?](https://www.aaai.org/ocs/index.php/WS/AAAIW18/paper/view/16578/15542)

<br/>
## Shallow-Wide Net

```
# for character-level
python3 char_shallownet.py

# for word-level
python3 word_shallownet.py
```

There are some arguments  

| arguments | default |note|
|:---:|:---:|:---:|
|data_dir|'../dataset/'||
|pos_file|'rt-polarity.pos'||
|neg_file|'rt-polarity.neg'||
|val_dir|None||
|val_pos_file|None||
|val_neg_file|None||
|model_dir|'./model/'||
|num_class|2||
|num_per_filters|char_shallownet : 700<br/> word_shallownet : 100||
|vocab|'vocab.pkl'| only in word-level<br/> if you set None, automatically download via gluonnlp|
|max_seq_len|char_shallownet : 1014<br/> word_shallownet: None||
|batch_size|128||
|seed|10||
|learning_rate|0.001||
|epochs|1||

<br/>

## Dense Net

```
# for character-level
python3 char_densenet.py

# for word-level
python3 word_densenet.py
```

There are some arguments  

| arguments | default |note|
|:---:|:---:|:---:|
|data_dir|'../dataset/'||
|pos_file|'rt-polarity.pos'||
|neg_file|'rt-polarity.neg'||
|val_dir|None||
|val_pos_file|None||
|val_neg_file|None||
|model_dir|'./model/'||
|num_class|2||
|vocab|'vocab.pkl'| only in word-level<br/> if you set None, automatically download via gluonnlp|
|max_seq_len|char_shallownet : 1014<br/> word_shallownet: None||
|batch_size|128||
|seed|10||
|learning_rate|0.001||
|epochs|1||
