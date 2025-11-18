# Server Website pipeline
  Very basic overview of the pipline


## Server 

### Reciving question data
    The server will be listening for any data. When the website sends 
    the data the server takes it. Then it get's the question and the 
    subreddits. 

#### What the script insigned the server do?
    We use the vector data baseses for the coresponding subreddits(that 
    that the website has send) that we have already prepeared. Then we use
    the question to get the most relevant posts from all databases(for now
    will get the first 10 most relevant question from every vector data base) 
    then from all posts we get the first number of posts the website 
    has given us. Then a script creats the ai summary.

#### Sending data back to website
    When the all steps are ready the server sends the data(the posts and the 
    ai summary) mabe with the question or something else to know for what 
    question the answers are not sure yet.  


## WebSite
    
### Data that will be used
    In the website many users have asked there questions and sugested what
    subreddits want to be used for answering the question. In the amdin panel
    the admin can remove or add different subrredit names. This subreddits
    will be used for answering the question.

### Sending the data 
    When the admin is sadesfied with question and the subreddits that have 
    been chosen, the admin can press a button 'Answer'. Then the data
    we described in the previus point will be send to the server. 

### Reciving response
    The website listens for any responses from the server. When there is
    one we take the data we make the question asnwered and we visualise
        1. The AI Summary
        2. All relevant reddit posts.
