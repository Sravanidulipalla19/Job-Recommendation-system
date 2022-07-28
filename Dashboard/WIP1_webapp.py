import streamlit as st
import pandas as pd

st.title("Job Recommender System")

st.header("*Problem Definition*")

st.write("""Create a recommender system that can recommend jobs based on the resume. The system will go over the resume and then search different job descriptions 
            that match the keywords on the resume.""")

st.header("*Problem Motivation*")

st.markdown("* Make it easier for individuals to find jobs hiring based on their skills.")
st.markdown("* Reduce the complications in the job application process.")
st.markdown("* Introduce people to other disciplines that require their skillset.")
st.markdown("* Help companies find individuals that match their job requirement.")

st.header("*Data/Dataset Description*")

st.markdown("* The dataset is from Indeed job board and we focus on entry level data relative job post.")
st.markdown("* We scrapped most relevant information from each job and stored them in a JSON format with following keys: company_name, role, location, salary_estimated, job_id, job_description,url")

from PIL import Image
image = Image.open("data details.png")
st.image(image)

st.header("*EDA Results*")

#############################
# EDA
#############################

import re
import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

# loading data
df = pd.read_json("data.json")
df = pd.concat([df,pd.read_json("data2.json")])
df.head()

# removing the rows with no job description
df = df.drop(['salary_estimated'], axis = 1)
df.dropna(inplace=True)

# Cleaning Job description data
df['job_description'] = df['job_description'].apply(lambda x: ''.join(''.join(x.split('\\n')).split('\n')))
df['job_description'] = df['job_description'].apply(lambda x: x.lower())
df['job_description'] = df['job_description'].apply(lambda x: re.sub('\w*\d\w*','', x))
df['job_description'] = df['job_description'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
df['job_description'] = df['job_description'].apply(lambda x: re.sub(' +',' ',x))

# Creating Document Term Matrix
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df['job_description'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index = df.index

# Importing wordcloud for plotting word clouds and textwrap for wrapping longer text
from wordcloud import WordCloud
from textwrap import wrap

# Function for generating word clouds
def plotwordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  fig = plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  st.balloons()
  st.pyplot(fig)
  

st.subheader("*Popular Jobs*")
# No of jobs plot
g1 = dict(df['role'].value_counts()[0:19])

fig = plt.figure(figsize =(10, 7))
plt.barh(list(g1.keys()), g1.values())
plt.title('Popular Data Jobs')
plt.xlabel('No of jobs')
plt.ylabel('Jobs')
plt.show()

st.balloons()
st.pyplot(fig)

# Word clouds
st.subheader("*Wordclouds of job descriptions of top 5 jobs*")

lemdf = df[['role','job_description']].groupby(by='role').agg(lambda x:' '.join(x)).loc[list(g1.keys())[0:5]]

lemdf['job_description'] = lemdf['job_description'].apply(lambda x: ' '.join(w for w in word_tokenize(x) if not w.lower() in stop_words))

data=cv.fit_transform(lemdf['job_description'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index = lemdf.index
# df_dtm

# Transposing document term matrix
df_dtm=df_dtm.transpose()

# Plotting word cloud for each product
for index,product in enumerate(df_dtm.columns):
  plotwordcloud(df_dtm[product].sort_values(ascending=False),product)

#############################


st.header("*Future Analysis*")

st.subheader("*Data Preprocessing*")

st.write("""Stop word removal - removes words that occur commonly across all documents (ex. is, and,
             your, that, etc.). These words provide no unique information to the text.""")

st.subheader("*Model Choice*")

st.markdown("1. Resume Parsing - extracts skills, experiences, degrees and company name from resume.")
st.markdown("2. Creating ngrams of skills - neighboring sequences of skills in a document.")
st.markdown("3. Vectorization using TF-IDF - assigns numerical values based on the word frequency.")
st.markdown("4. Matching resume to job description - using KNN and cosine similarity.")

# st.header("*Challenges/Future Directions*")

st.header("*Expected Output*")

st.write("A set of Job recommendations that matches one’s resume based on the skills extracted from the resume. ")

st.header("*Team Progress*")

st.subheader("*Job scraping*")

st.markdown("* Able to scrap job information from Indeed job board.")
st.markdown("* Initial progress on linkedin web scraper creation as an alternative data source.")

st.subheader("*Model selection*")

st.markdown("* Steps for preprocessing resume and job description")
st.markdown("* Model choice for resume parser and matching resume to jobs.")

st.subheader("*Dashboard*")

st.markdown("* Created a webapp with streamlit for this presentation.")
st.markdown("* Completed and integrated the Exploratory Data Analysis (EDA) into webapp.")

st.header("*Roles & Responsibilities*")

st.subheader("Team leader: Shun An Chang")
st.write("#### Responsibilities:")
st.markdown("* Organize the team and host the meeting")
st.markdown("* Make sure every task done before the due date.")

st.subheader("Team Secretary: Chidubem Okorozo")
st.write("#### Responsibilities:")
st.markdown("* Take notes for every meeting.")
st.markdown("* Record attendance of every meeting.")

st.subheader("Team Git manager: Sravani Dulipalla")
st.write("#### Responsibilities:")
st.markdown("* Manage the Git Repository")
st.markdown("* Make sure every team member use the git properly.")

st.subheader("Team Task manager: Hemanth Talla")
st.write("#### Responsibilities:")
st.markdown("* Make sure every team member finish their task on time.")
st.markdown("* Make sure every subgroup to use the task board properly.")

st.subheader("Web Scraping SubGroup: Harsh Nisar, Shun An Chang, Hemanth Talla")
st.write("#### Responsibility:")
st.markdown("* Create a robust scheduled web scraping program that can get job information and store them in cloud base data lake.")
# st.markdown("* ")

st.subheader("NNL model SubGroup: Chidubem Okorozo, Saad Azim Vaibhav Patel")
st.write("#### Responsibility:")
st.markdown("* Create a NNL model to produce matching metric from user’s resume and job description.")
# st.markdown("* ")

st.subheader("DashBoard SubGroup: Sravani Dulipalla, Bhargav Singuluri")
st.write("#### Responsibilities:")
st.markdown("* Create a Web based Dashboard allow user to interact such as uploading resume.")
st.markdown("* Show users the most relative jobs for them based on their resume and EDA.")

st.header("*Challenges/Future Directions*")

st.markdown("* Setting up chrome drivers.")
st.markdown("* Creating ngrams model of skills and matching the skills to the resume.")
st.markdown("* Deploy the application on cloud based environment.")

st.header("*Timeline for Future Plans*")

st.markdown("1. 7/4 - 7/11 - Complete Matching Model, Deploy scrapping program on cloud.")
st.markdown("2. 7/11 - 7/18 - Creating the Dashboard. Integrating Model results with the dashboard.")
st.markdown("3. 7/18 - 7/25 - Solving minor issues and optimizing the model.")

