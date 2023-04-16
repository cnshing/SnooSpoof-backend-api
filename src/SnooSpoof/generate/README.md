##DAtaset Format
A typical submission example from the dataset is as follows:

```
is_original_content: None
over_18: None
post: comment
subreddit: Genshin_Impact
prompt: What do you think about Genshin Impact?
response: I think its great. It's a fun and addicting game that can be played anywhere. I personally like how...
```

In other words, it is simply line-delimited attributes concanetated as one, single text. The main advantage from manufacturing our examples to lead generation is that any arbitrary model can now be inserted to generate text. With enough examples, any model should be able to consistenly create text that strictly follows this format.  
