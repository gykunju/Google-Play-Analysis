import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud


st.set_page_config(
    page_icon=':bar_chart:'
)
st.write('''
# GOOGLE PLAY STORE APPS ANALYSIS   
''')


### Loading the App Data from pickle
st.write('''
#### Data sets:
         ''')

with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

### Loading the Reviews Data from pickle
with open('reviews.pkl', 'rb') as file:
    reviews = pickle.load(file)

with open('full_data.pkl', 'rb') as file:
    full_data = pickle.load(file)

st.expander('Data').dataframe(data)
st.expander('Reviews').dataframe(reviews)
st.expander('full_data').dataframe(full_data)


st.write('''
#### Summary
         ''')
col1, col2, col3 = st.columns(3)
col1.metric('Apps', data.count()[0])
col2.metric('Categories', data.groupby('Category').count().count()[0])
col3.metric('Max Installs', data.sort_values(by='Installs', ascending=False).head(1).loc[:,'Installs'])

col1, col2 = st.columns(2)
col1.metric('Max Size', data.sort_values(by='Size', ascending=False).head(1).loc[:,'Size'])

### Top Categories Based on Installs
st.write('''
## App Distribution Based on Installs
         ''')
sorted_categories = data.loc[:,['Installs', 'Category']].groupby('Category').sum().reset_index()

merge_max = data.loc[:, ['Installs', 'Category']].groupby('Category').max().merge(sorted_categories, on='Category', suffixes=('_max', '_total'))
complete_merge = merge_max.merge(data.loc[:,['Installs', 'Category']].groupby('Category').min(), on='Category')
complete_merge = complete_merge.rename(columns={
    'Installs_max': 'Max',
    'Installs_total': 'Total',
    'Installs': 'Min'
})
complete_merge = complete_merge.sort_values(by='Total', ascending=False)

categories = complete_merge['Category'].to_list()

width = 0.8
fig, ax = plt.subplots(figsize=(15,13))
X = np.arange(len(complete_merge['Category']))

ax.barh(complete_merge['Category'], complete_merge['Total'], ec='black',zorder = 3, color='teal')

# ax.set_xticks(ticks= X , labels=categories, rotation=90)

# plt.tick_params(axis='both',labelsize=13,left=False,)

ax.grid(axis='x')
ax.invert_yaxis()
plt.ylabel('Installs', fontsize=14)
plt.xlabel('Categories', fontsize=14)
plt.title('Top Ten Categories Based on Installs', fontweight='bold', fontsize='22', pad=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
sns.despine()
st.pyplot(fig)

st.write('''
#### Insights
         
- Apps in the Game Category have the most installs

- Apps in the Events Category have the least Installs
         ''')



### GROUPING APP CATEGORIES IN TERMS OF SIZE

st.subheader('Category Distribution Size-wise')


grouped_data = data.loc[:,['Size', 'Category']]

fig , ax = plt.subplots(figsize=(18,20))
ticks = np.arange(len(grouped_data.groupby('Category')))

sns.boxplot(y=grouped_data['Category'], x=grouped_data['Size'], ax=ax)
plt.tight_layout()
plt.tick_params(axis='both',labelsize=13,direction='out',left=False,bottom=False)

plt.title('Category Distribution in Terms of Size', fontweight='bold', fontsize='22', pad=20)
sns.despine()

st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- The median app size varies significantly across categories.         
         
- Categories like Game, Sports, Family and Travel and local have a wide variability in their app sizes
         
- Categories like Events, Weather, House and home and comics have a small spread indicating consistency in their app sizes
         ''')

### DISTRIBUTION OF REVIEWS
st.subheader('Sentiment Distribution of Reviews')
fig, ax = plt.subplots(figsize=(5,5))

sentiments = reviews.groupby('Sentiment').count().reset_index()
plt.pie(sentiments['App'], labels=sentiments['Sentiment'], autopct='%1.1f%%', wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' })


st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- Majority of the reviews(64.1%) are positive while there is a close margin between neutral reviews(13.8%) and negative reviews(22.1%)
         ''')

### TOP APPS BASED ON REVIEWS

st.subheader('Top Reviewed Apps')
x = data.sort_values(by='Reviews', ascending=False).iloc[:20,:]['App']
y = data.sort_values(by='Reviews', ascending=False).iloc[:20,:]['Reviews']

fig, ax = plt.subplots(figsize=(15,10),)

ax.barh(x,y, color='teal', alpha=0.9, ec='black')
ax.invert_yaxis()

ax.grid(axis='x')
plt.title('Top Reviewed Apps in Google Play Store', fontweight='bold', fontsize=15)
plt.ylabel('App', fontsize=12)
plt.xlabel('Reviews', fontsize=12)
sns.despine(left=True)
st.pyplot(fig)


### Distribution of Positive Reviews

st.subheader('Distribution Of Postive Reviews')

# Geting Top 10 Positive Reviews
positive_reviews = full_data[full_data.Sentiment == 'Positive'].loc[:,['App', 'Sentiment']].groupby('App').count().reset_index().sort_values(by='Sentiment', ascending=False)

# Getting the subjectivity of the reviews
subjectivity = full_data[(full_data.Sentiment == "Positive") & (full_data.Sentiment_Subjectivity >= 0.5)].loc[:,['App', 'Sentiment_Subjectivity']].groupby('App').count().reset_index()

# Merging the subjectivity and the positive_reviews
positive_reviews_subjectivity = pd.merge(positive_reviews, subjectivity, on='App', how='left')

col1, col2 = st.columns((7,3))
col1.metric('Most Positive Reviews', positive_reviews_subjectivity['App'][0], int(positive_reviews_subjectivity['Sentiment'][0]))
col2.metric('Least Positive Reviews', positive_reviews_subjectivity['App'][804], int(positive_reviews_subjectivity['Sentiment'][804]))


### PLOT OF TOP 10 MOST POSITIVELY REVIEWED APPS
fig, ax = plt.subplots(figsize=(20,12))

x = positive_reviews_subjectivity['App'].head(10)
y= positive_reviews_subjectivity['Sentiment'].head(10)
y2 = positive_reviews_subjectivity['Sentiment_Subjectivity'].head(10)
x_list = np.arange(len(x))
labels = x.to_list()
width = 0.25

ax.bar(x_list - width, y, width=width, color='orange', edgecolor='black', label='Positive Reviews')
ax.bar(x_list, y2, width=width, color='teal', edgecolor='black', label='Subjective Reviews')
ax.bar(x_list + width, y - y2, width=width, color='yellow', edgecolor='black', label='Objective Reviews')

plt.title('Distribution of Positive Reviews (Top 10)', fontweight='bold', fontsize=15)
plt.xlabel('App')
plt.ylabel('Count')
plt.grid(axis='x')


ax.set_xticks(ticks=x_list)
ax.set_xticklabels(labels, rotation=90)

sns.despine()

plt.legend()
st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- Duolingo has the most positive reviews with an almost equal split between subjectivity and objectivity
         
- The positive reveiws in the dataset exhibit high levels of subjectivity
         ''')


### PLOT OF BOTTOM 10 MOST POSITIVELY REVIEWED APPS

fig, ax = plt.subplots(figsize=(20,12))

x = positive_reviews_subjectivity['App'].tail(10)
y= positive_reviews_subjectivity['Sentiment'].tail(10)
y2 = positive_reviews_subjectivity['Sentiment_Subjectivity'].tail(10)
x_list = np.arange(len(x))
labels = x.to_list()
width = 0.25

ax.bar(x_list - width, y, width=width, color='orange', edgecolor='black', label='Positive Reviews')
ax.bar(x_list, y2, width=width, color='teal', edgecolor='black', label='Subjective Reviews')
ax.bar(x_list + width, y - y2, width=width, color='yellow', edgecolor='black', label='Objective Reviews')

plt.title('Distribution of Positive Reviews (Bottom 10)', fontweight='bold', fontsize=15)
plt.xlabel('App')
plt.ylabel('Count')
plt.grid(axis='x')


ax.set_xticks(ticks=x_list)
ax.set_xticklabels(labels, rotation=90)

sns.despine()

plt.legend()
st.pyplot(fig)   

### DUOLINGO ANALYSIS
st.write('''
## DUOLINGO
         ''')

duolingo = full_data[full_data.App == 'Duolingo: Learn Languages Free'].reset_index()
col1, col2, col3 = st.columns(3)
col1.metric('Installs' ,duolingo.loc[:,'Installs'][0])
col2.metric('Size', duolingo.loc[0,'Size'])
col3.metric('Version', duolingo.loc[0,'Current Ver'])

col1, col2 = st.columns(2)
col1.metric('Price', duolingo.loc[0,'Price'])

col1, col2 = st.columns(2)

### pie chart
fig, ax = plt.subplots(figsize=(4,4))

duolingo = full_data[full_data.App == 'Duolingo: Learn Languages Free'].reset_index()
duolingo_reviews = duolingo.groupby('Sentiment').count().reset_index().loc[:,'App']
duolingo_reviews.index = ['Negative', 'Neutral', 'Positive']
plt.pie(duolingo_reviews, labels=duolingo_reviews.index, autopct='%1.1f%%', wedgeprops = { 'linewidth' : 4, 'edgecolor' : 'white' })

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

with col1:
    st.write('''
#### Distribution of Sentiments
    ''')
    st.pyplot(fig)

### WORDCLOUD OF THE MOST REPEATED WORD IN DUOLINGO REVIEWS

fig, ax = plt.subplots(figsize=(9,9))
text = ','.join(duolingo.loc[:,'Translated_Review'].to_list())

wordcloud = WordCloud(width=400, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis('off')

with col2:
    st.write('''
#### Most Frequent Words
    ''')
    st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- Has 83% of its reviews as positive
         
- Majority of the reviews involve learning languages 
         ''')


### HOMEWORK ANALYSIS

st.write('''
## HOMEWORK
         ''')

homework = full_data[full_data.App == 'HomeWork'].reset_index()
col1, col2, col3 = st.columns(3)
col1.metric('Installs' ,homework.loc[:,'Installs'][0])
col2.metric('Size', homework.loc[0,'Size'])
col3.metric('Version', homework.loc[0,'Current Ver'])

col1, col2 = st.columns(2)
col1.metric('Price', homework.loc[0,'Price'])

col1, col2 = st.columns(2)

### pie chart


fig, ax = plt.subplots(figsize=(4,4))

homework = full_data[full_data.App == 'HomeWork'].reset_index()
homework_reviews = homework.groupby('Sentiment').count().reset_index().loc[:,'App']
homework_reviews.index = ['Positive']

plt.pie(homework_reviews, labels=homework_reviews.index, autopct='%1.1f%%', wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

with col1:
    st.write('''
#### Distribution of Sentiments
    ''')
    st.pyplot(fig)


### WORDCLOUD OF THE MOST REPEATED WORD IN HOMEWORK REVIEWS

fig, ax = plt.subplots(figsize=(9,9))

text = ''.join(homework[homework.Translated_Review.notna()].loc[:,'Translated_Review'].to_list())
wordcloud = WordCloud(width=400, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis('off')

with col2:
    st.write('''
#### Most Frequent Words
    ''')
    st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- Has one sole review that is positive
         
- The most frequently appearing words are 'homework', 'best' and 'world'
         ''')

### DISTRIBUTION OF NEGATIVE REVIEWS

st.subheader('Distribution Of Negative Reviews')

# Geting Top 10 Negative Reviews
negative_reviews = full_data[full_data.Sentiment == 'Negative'].loc[:,['App', 'Sentiment']].groupby('App').count().reset_index().sort_values(by='Sentiment', ascending=False)

# Getting the subjectivity of the reviews
subjectivity = full_data[(full_data.Sentiment == "Negative") & (full_data.Sentiment_Subjectivity >= 0.5)].loc[:,['App', 'Sentiment_Subjectivity']].groupby('App').count().reset_index()

# Merging the subjectivity and the negative_reviews
negative_reviews_subjectivity = pd.merge(negative_reviews, subjectivity, on='App', how='left')

col1, col2 = st.columns((7,3))
col1.metric('Most Positive Reviews', negative_reviews_subjectivity['App'][0], int(negative_reviews_subjectivity['Sentiment'][0]))
col2.metric('Least Positive Reviews', negative_reviews_subjectivity['App'][739], int(negative_reviews_subjectivity['Sentiment'][739]))

### PLOT OF TOP 10 MOST NEGATIVELY REVIEWED APPS

fig, ax = plt.subplots(figsize=(15,10))

x = negative_reviews_subjectivity['App'].head(10)
y= negative_reviews_subjectivity['Sentiment'].head(10)
y2 = negative_reviews_subjectivity['Sentiment_Subjectivity'].head(10)
x_list = np.arange(len(x))
labels = x.to_list()
width = 0.25


ax.bar(x_list - width, y, ec='black', width=width, color='orange', alpha=0.9, label='Negative Reviews')
ax.bar(x_list, y2, ec='black', width=width, color='teal',alpha=0.6, label='Subjective Reviews')
ax.bar(x_list + width, y-y2,  ec='black', width=width, color='red', alpha=0.5, label='Objective Reviews')

plt.title('Distribution of Negative Reviews (Top 10)', fontweight='bold', fontsize=15)
plt.xlabel('App')
plt.ylabel('Count')
plt.grid(axis='x')

ax.set_xticks(ticks=x_list)
ax.set_xticklabels(labels, rotation=90)
sns.despine()

plt.legend()

st.pyplot(fig)


### PLOT OF BOTTOM 10 MOST NEGATIVELY REVIEWED APPS

fig, ax = plt.subplots(figsize=(15,10))

x = negative_reviews_subjectivity['App'].tail(10)
y= negative_reviews_subjectivity['Sentiment'].tail(10)
y2 = negative_reviews_subjectivity['Sentiment_Subjectivity'].tail(10)
x_list = np.arange(len(x))
labels = x.to_list()
width = 0.25


ax.bar(x_list - width, y, ec='black', width=width, color='orange', alpha=0.9, label='Negative Reviews')
ax.bar(x_list, y2, ec='black', width=width, color='teal',alpha=0.6, label='Subjective Reviews')
ax.bar(x_list + width, y-y2,  ec='black', width=width, color='red', alpha=0.5, label='Objective Reviews')

plt.title('Distribution of Negative Reviews (Bottom 10)', fontweight='bold', fontsize=15)
plt.xlabel('App')
plt.ylabel('Count')
plt.grid(axis='x')

ax.set_xticks(ticks=x_list)
ax.set_xticklabels(labels, rotation=90)
sns.despine()

plt.legend()

st.pyplot(fig)


### ANGRY BIRDS ANALYSIS

st.write('''
## ANGRY BIRDS
         ''')

angry_birds = full_data[full_data.App == 'Angry Birds Classic'].reset_index()
col1, col2, col3 = st.columns(3)
col1.metric('Installs' ,angry_birds.loc[:,'Installs'][0])
col2.metric('Size', angry_birds.loc[0,'Size'])
col3.metric('Version', angry_birds.loc[0,'Current Ver'])


col1, col2 = st.columns(2)
col1.metric('Price', angry_birds.loc[0,'Price'])

col1, col2 = st.columns(2)

### pie chart

fig, ax = plt.subplots(figsize=(4,4))

angry_birds_reviews = angry_birds.groupby('Sentiment').count().reset_index().loc[:,'App']
angry_birds_reviews.index = ['Negative', 'Neutral', 'Positive']

plt.pie(angry_birds_reviews, labels=angry_birds_reviews.index, autopct='%1.1f%%', wedgeprops = { 'linewidth' : 1.5, 'edgecolor' : 'white' })

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

with col1:
    st.write('''
#### Distribution of Sentiments
    ''')
    st.pyplot(fig)

### WORDCLOUD OF THE MOST REPEATED WORD IN ANGRY BIRDS REVIEWS

fig, ax = plt.subplots(figsize=(9,9))

text = ','.join(angry_birds[angry_birds.Translated_Review.notna()].loc[:,'Translated_Review'].to_list())
wordcloud = WordCloud(width=400, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis('off')

with col2:
    st.write('''
#### Most Frequent Words
    ''')
    st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- It has a fair split between Positive and Negative Sentiments
         
- 'game', 'play' and 'ad' are the most frequently used words    

- Having the largest number of negative reviews does not cripple its installation numbers
         ''')

### 2018 Emoji Keyboard


st.write('''
## 2018 EMOJI KEYBOARD
         ''')

emoji_keyboard = full_data[full_data.App == '2018Emoji Keyboard 😂 Emoticons Lite -sticker&gif'].reset_index()
col1, col2, col3 = st.columns(3)
col1.metric('Installs' ,emoji_keyboard.loc[:,'Installs'][0])
col2.metric('Size', emoji_keyboard.loc[0,'Size'])
col3.metric('Version', emoji_keyboard.loc[0,'Current Ver'])

col1, col2 = st.columns(2)
col1.metric('Price', emoji_keyboard.loc[0,'Price'])

col1, col2 = st.columns(2)

### pie chart

fig, ax = plt.subplots(figsize=(4,4))

emoji_keyboard_reviews = emoji_keyboard.groupby('Sentiment').count().reset_index().loc[:,'App']
emoji_keyboard_reviews.index = ['Negative', 'Neutral', 'Positive']

plt.pie(emoji_keyboard_reviews, labels=emoji_keyboard_reviews.index, autopct='%1.1f%%', wedgeprops = { 'linewidth' : 1.5, 'edgecolor' : 'white' })

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

with col1:
    st.write('''
#### Distribution of Sentiments
    ''')
    st.pyplot(fig)


### WORDCLOUD OF THE MOST REPEATED WORD IN 2018Emoji REVIEWS

fig, ax = plt.subplots(figsize=(9,9))

text = ','.join(emoji_keyboard[emoji_keyboard.Translated_Review.notna()].loc[:,'Translated_Review'].to_list())
wordcloud = WordCloud(width=400, height=400).generate(text)

plt.imshow(wordcloud)
plt.axis('off')

with col2:
    st.write('''
#### Most Frequent Words
    ''')
    st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- 2018 Keyboard has 78% of its reviews as positive
         
- 'Good', 'emoji' and 'keyboard' are the most frequently used words in the reviews
         
- Despite it having the least amount of negative reviews its installation numbers are less tha the most negative indicating the influence of another factor  
         ''')


### RELATIONSHIPS

### RELATIONSHIP BETWEEN SIZE AND PRICE

st.subheader('Relationship  Between Size and Price')
fig , ax = plt.subplots(figsize=(10,10)) 
sns.scatterplot(data=data, x='Size', y='Price')

st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- Majority of The Apps are Priced below $50
         
- The Price does not have a significant effect on the size
         ''')

### RELATIONSHIP BETWEEN RATINGS AND INSTALLS

st.subheader('Relationship Between Ratings and Installs')
fig, ax = plt.subplots(figsize=(15,10))

sns.scatterplot(x=data['Rating'], y=data['Installs'])
plt.title('Relationship Between Rating and Installs', fontweight='bold', fontsize=15)

st.pyplot(fig)

st.write('''
#### INSIGHTS
         
- The number of Installs increases with the App Rating

         ''')

### RELATIONSHIP BETWEEN PRICE AND INSTALLS
