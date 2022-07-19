import numpy as np
import pandas as pd
import streamlit as st
import PyPDF2
import os
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pyresparser import ResumeParser
import re
import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#############################
# Importing Data
#############################

import boto3
import json
import sys
s3 = boto3.client('s3')

BUCKET_NAME = 'job-recommendation-system-cleanse-useast-1-69522247-dev'

# get jobs information from single json file
def get_jobs_from_key(bucket_name, key):

    response = s3.get_object(
        Bucket=bucket_name,
        Key=key,
    )
    jobs_list = json.loads(response['Body'].read().decode('utf-8'))
    return jobs_list

# accumulate jobs information between certain periods
def get_jobs_from_prefix(bucket_name, prefix):
    return_jobs_list = []
    response = s3.list_objects(
        Bucket='job-recommendation-system-cleanse-useast-1-69522247-dev',
        Prefix='2022/07/',
    )

    for json_file in response['Contents']:
        key = json_file['Key']
        jobs_list = get_jobs_from_key(BUCKET_NAME, key)
        if len(jobs_list) != 0:
            return_jobs_list.extend(jobs_list)

    return return_jobs_list


for job in get_jobs_from_prefix(BUCKET_NAME, '2022/07/'):
    print(job)

#############################
# Model
#############################

st.title("Job Recommendation System")

# File uploader
st.subheader("*Upload a resume PDF file*")
fl = st.file_uploader("")

# loading data
df = pd.DataFrame(get_jobs_from_prefix(BUCKET_NAME, '2022/07/'))

# removing the rows with no job description
df = df.drop(['salary_estimated'], axis = 1)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Cleaning Job description data
df['job_description'] = df['job_description'].apply(lambda x: ''.join(''.join(x.split('\\n')).split('\n')))
df['job_description'] = df['job_description'].apply(lambda x: x.lower())
df['job_description'] = df['job_description'].apply(lambda x: re.sub('\w*\d\w*','', x))
df['job_description'] = df['job_description'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
df['job_description'] = df['job_description'].apply(lambda x: re.sub(' +',' ',x))

if fl:
    with open(os.path.join(os.getcwd(),fl.name),"wb") as f:
            f.write(fl.getbuffer())

    # Resume parser
    data = ResumeParser(fl.name).get_extracted_data()
    resume = data['skills']
    skills=[]
    skills.append(' '.join(word for word in resume))
    # st.write(skills)
    
    # Documents list with skills and job descriptions
    documents = list(skills)
    for i in list(df['job_description']):
        documents.append(i)
    # st.write("doc[0]")
    # st.write(documents[0])

    # Cosine similarity model
    TfidfVec = TfidfVectorizer(stop_words='english')
    def cos_similarity(textlist):
        tfidf = TfidfVec.fit_transform(textlist)
        return (tfidf * tfidf.T).toarray()

    # Results
    final = cos_similarity(documents)

    dist = final[0][1:]
    keys = range(1,df.shape[0]+1)
    dist_dict = dict(zip(keys,dist))

    dist_dict_sort = sorted(dist_dict, key = dist_dict.get, reverse=True)
    # st.write(dist_dict_sort[0:10])

    st.subheader("*Jobs related to the resume*")
    
    st.write(df.iloc[dist_dict_sort[0:10]])

#############################
# EDA
#############################

st.header("*EDA*")

stop_words = set(stopwords.words('english'))

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
