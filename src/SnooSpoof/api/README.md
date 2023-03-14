
Parameters
cUrl request:
```bash
curl -X 'GET' \
  'http://API_URL:API_PORT/generate/?username=Spez&comments_only=false&is_original_content=false&over_18=false' \
  -H 'accept: application/json'
 ```
 
 with optional prompt: "I'm a big fan of reddit"
 ```bash
curl -X 'GET' \
  'http://API_URL:API_PORT/generate/?username=Spez&prompt=I%27m%20a%20big%20fan%20of%20reddit&comments_only=false&is_original_content=false&over_18=false' \
  -H 'accept: application/json'
 ```
 
 Response:
 ```json
 {
  "subreddit": "reddit.com",
  "prompt": "I'm a big fan of reddit, and I've been using it for a while now. I think it's a great way to get people to share their experiences and ideas.",
  "response": "\n\nI'm not sure what you mean by \"sharing your experiences\" or what it means to you.   I don't know if it is a good idea to have a bunch of people share your experience with you, but I do think that it would be a better idea if we could share it with people who are interested in learning more about reddit and what makes it tick. It's also a way for people like you to be able to see what's going on in the world around you and to know what your favorite subreddits are. If you're interested, you can find out more here"
}
```
Request body:

username: string => The username the text should make an impression of.
subreddit: Optional[string] => Insert subreddit into initial text to bias text generation
prompt: Optional[string] => Insert an initial prompt into text to bias text generation
comments_only: boolean | none => Mark initial text with boolean that is always true with comment_reply posts to bias text generation
is_original_content: boolean | null => Same as above
over_18: boolean | null  => Same as above
