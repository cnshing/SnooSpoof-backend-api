## Dataset Format
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

## Scrapper

The scrapper component extends existing PRAW functionality by allowing declarative searches for comments, submissions, and parents. 

For example, to get the comment body that contains the letter "a" for Spez, do the following:

```
comments = self.scrapper.search_comments(author="Spez", tags="body", filter_fn=lambda comment: 'a' in comment['body'])
```
The details on what exactly is being collected is in [api/generate.py](https://github.com/cnshing/SnooSpoof-backend-api/blob/main/src/SnooSpoof/api/generate.py) in the _fetch_data function. 


