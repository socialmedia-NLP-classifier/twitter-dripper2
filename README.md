# Twitter-Classifier

Take in a post from twitter/facebook/ and other social-media platforms and train the model to classify the text with an “importance scale”.  If the input consists of images, we add a tag with appropriate detail to the text input and hope the classifier rates the image tag higher.  Example tag [IMAGE, 1080x400, name of pic].
After we classify a text/post we can filter the collection of tweets and display the posts that are worth reading.
Input: text and other meta-data of the post
Output: either a Yes vs No and a numerical scale for how important the post is.


# Table of Content:
- [twitter-classifier](#twitter-classifier)
- [Table of Content:](#table-of-content)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Flask App](#flask-app)
  - [Hosted on Heroku](#hosted-on-heroku)
- [Building an Flask App](#building-an-flask-app)
      - [Flask Documentation](#flask-documentation)
      - [Bootstrap Documentation](#bootstrap-documentation)
- [Extra Credit:  Deploying your app to Heroku.](#extra-credit--deploying-your-app-to-heroku)
    - [STEP 1: Create a github repo with your webapp files.](#step-1-create-a-github-repo-with-your-webapp-files)
    - [STEP 2: Create and push files to Heroku.](#step-2-create-and-push-files-to-heroku)
    - [Making changes to your app.](#making-changes-to-your-app)


## Dependencies

https://stackoverflow.com/questions/8165028/newbie-in-heroku-error-when-push-my-app-to-heroku/8171155#8171155


## Datasets
- https://www.kaggle.com/mtesconi/twitter-deep-fake-text
- https://www.kaggle.com/phoeyleeteh/1970-vs-2017-profanity-curse-swearing-words
- https://www.kaggle.com/davidmartngutirrez/twitter-bots-accounts
- https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Twitter_Data.csv


## Flask App

[//]: # what exactly was done for the flask app?

For the appearance portion: We used the Bootstrap Framework to provide the Python Flask App with a more appealing user interface
- The navbar allows the user to navigate to the main page, the visualized results and a page on the creators
- On the visualized results page, core components of Bootstrap were utilized including the jumbotron and the carousel to provide
 easy access performance results of the model


## Hosted on Heroku

https://twitter-dripper2.herokuapp.com/


# Building an Flask App

#### [Flask Documentation](https://flask.palletsprojects.com/en/1.1.x/quickstart/)
#### [Bootstrap Documentation](https://getbootstrap.com/docs/4.3/components/alerts/)
[Bootstrap Themes](https://getbootstrap.com/docs/4.0/examples/)

# Extra Credit:  Deploying your app to Heroku.
Once you have a flask app running locally on your machine already, you can then try to make a live website that anyone can visit by depolying it to Heroku.

These instructions were adapted from [this tutorial blog](https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7).

1. Sign up for a free Heroku account at https://signup.heroku.com/signup/dc
2. You should all already have git installed, so you can skip this step. Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed.
3. For Mac Users: [Install Homebrew](https://brew.sh/) if you dont have it already. (Window users can skip this step.)
4. Install the [Heroku CLI tool](https://devcenter.heroku.com/articles/heroku-cli#download-and-install).  Mac users need homebrew to install Heroku, Window users dont need it to install Heroku.

___

### STEP 1: Create a github repo with your webapp files.
__(You should already have this step completed, but if you dont, you need to add your files to a github repo.)__
1. Create a github repo for your app.
2. Clone that repo to your local machine.
3. Git add, commit, and push the webapp files to your repo.
Now you should have a github repo that contains all the files you need for your web app, now we need to link, push, and deploy them onto heroku. Your github repo should look [something like this](https://github.com/zd123/zd-flask-app).

___
### STEP 2: Create and push files to Heroku.
1. Make sure you have followed the instructions above for installing the Heroku CLI and git to your machine first.
2. From your github repo folder, in your terminal, type in `heroku login`.  Follow the login instructions.
3. Next create your app using `heroku create your-webapp-name`.  Replace 'your-webapp-name' with the app name you would like to use.  It will be a part of your final URL.
4. Then, add your git to heroku by using `heroku git:remote -a your-webapp-name`.  Again, replace 'your-webapp-name' with the name you used above to create your app.
5. Finally, push your files to heroku by using `git push heroku HEAD:master`
After you push your files to heroku, it should spit out a link to your webapp, it will look something like, `https://web-app-name.herokuapp.com/`

___
### Making changes to your app.
Say you've updated your webapp files now want to update your hosted webapp.
1. Add, commit, and push all the changes to your github repo.
2. Run the `git push heroku HEAD:master` command.


