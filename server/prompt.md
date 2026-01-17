When the server has the subreddits and the question. 
We want to: 
    1. Check if all subreddits recived exists in the SubReddit folder in the root of the project. For every non existing one we want a msg printed and the program to continue to run
    2. Then we want to use the question and get all the closests posts using the vector database. For every subreddit the scraped posts, comments and the generated vector database
        will be stored in a folder with the name of the subreddit stored in the SubReddit folder in the root of the project.We want to get the first 10 most relevant posts and from 
        every subreddit that has been send to the server. And from all of this posts get the first 10.
    3. Then generate an AI Summary on all posts that are get use the code below.

I want when this things are done to print the ai sammary, and the posts.
If you have any question ask me.Think extra hard.
```
for subreddit in subreddits:
    vec_db = VectorRetriever(directory_from_subreddit(subreddit))
    new_candidates = vec_db.retrieve(query, topk=k)
    candidates.extend(new_candidates)

best_candidates = candidates.sort(lambda x: -x.score)[:k]

context = build_context(best_candidates)

answer = maybe_xai_answer(SYSTEM_PROMPT, query, context)

return answer
```
