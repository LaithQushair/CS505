import pandas as pd
import nltk
nltk.data.path.append('/usr4/ece601/lqushair/.conda/envs/EC523_model/nltk_data')  # e.g., your home directory
nltk.download('punkt', download_dir='/usr4/ece601/lqushair/.conda/envs/EC523_model/nltk_data')
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('/projectnb/cs505aw/projects/tonality/evidence-2015.csv')
df2 = pd.read_csv('/projectnb/cs505aw/projects/tonality/evidence-2018.csv')
df3 = pd.read_csv('/projectnb/cs505aw/projects/tonality/evidence-2020.csv')
df4 = pd.read_csv('/projectnb/cs505aw/projects/tonality/evidence-2022.csv')
df['summary'].to_csv('2015.csv', index=False)
df2['summary'].to_csv('2018.csv', index=False)
df3['summary'].to_csv('2020.csv', index=False)
df4['summary'].to_csv('2022.csv', index=False)
