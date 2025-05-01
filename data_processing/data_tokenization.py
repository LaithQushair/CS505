from nltk.sentiment.vader import SentimentIntensityAnalyzer
df = pd.read_csv('/content/2015.csv', on_bad_lines='warn', engine='python')
text = df['summary']
text_list = []
for i in df['summary']:
    if type(i) is str:
        if len(i) > 150:
            text_list.append(i)
individual_sentences = []
for i in text_list:
    sentences = sent_tokenize(i)
    for j in sentences:
        individual_sentences.append(j)
arg_sentences = []
persuasive_sentences = []
sid = SentimentIntensityAnalyzer()
for i in individual_sentences:
  sentiment = sid.polarity_scores(i)
  if sentiment['neu'] > 0.6:
      arg_sentences.append(i)
  else:
      persuasive_sentences.append(i)

  try:
    with open('/content/02. College Algebra Autor Jay Abramson.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text2 = ''
        for page_num in range(len(reader.pages)-2):
            page = reader.pages[page_num]
            text2 += page.extract_text()
except FileNotFoundError:
    print("The file was not found.")
try:
    with open('/content/s41598-025-97208-8.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)-2):
            page = reader.pages[page_num]
            text += page.extract_text()
except FileNotFoundError:
    print("The file was not found.")

info_text = text + text2
individual_sentences = []
sentences = sent_tokenize(info_text)
for j in sentences:
  individual_sentences.append(j)

x_data = individual_sentences + arg_sentences + persuasive_sentences
y_data = [0] * len(individual_sentences) + [1] * len(arg_sentences) + [2] * len(persuasive_sentences) #0 = info, 1 = arg, 2 = persuasive
