import matplotlib.pyplot as plt
import gradio as gr
import nltk
nltk.download('all')
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import pickle

class TonalityLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

with open("vocab.pkl", 'rb') as file:
    vocab = pickle.load(file)

model = torch.load("full_model.pt", weights_only=False, map_location=torch.device('cpu'))
lemmatizer = WordNetLemmatizer()

max_length = 40

def classify_text(text):
    sentences = sent_tokenize(text)
    output = ""
    for sentence in sentences:
      tokens = word_tokenize(sentence)
      seq = []
      for word in tokens:
        new_word = lemmatizer.lemmatize(word.lower())
        seq.append(vocab.get(new_word, vocab["<UNK>"]))
      seq = seq[:max_length] + [0]*(max_length - len(seq))
      input_tensor = torch.tensor([seq], dtype=torch.long)
      with torch.no_grad():
        pred = model(input_tensor)
        label = torch.argmax(pred, dim=1).item()

        tone = ["Informative", "Argumentative", "Persuasive"][label]
        output += f"[{tone}] {sentence}\n"

    return output.strip() #0 = info, 1 = arg, 2 = persuasive

def gradio_classify(text):
    try:
        result = classify_text(text)
        color_map = {
            "Informative": "#00bfff",
            "Argumentative": "#DC143C",
            "Persuasive": '#FFC000'
        }
        counts = {"Informative": 0, "Argumentative": 0, "Persuasive": 0}
        formatted_lines = []
        for line in result.split("\n"):
            tone_part, sentence = line.split("] ", 1)
            tone = tone_part[1:]
            counts[tone] += 1

            color = color_map[tone]
            formatted_line = (
                f"<span style='color: {color}; font-weight: 500; margin: 4px 0; display: block;'>"
                f"▌{tone}</span> "
                f"<span style='color: {color}80;'>{sentence}</span>"
            )
            formatted_lines.append(formatted_line)

        #
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [k for k, v in counts.items() if v > 0]
        sizes = [v for v in counts.values() if v > 0]
        colors = [color_map[label] for label in labels]

        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%2.1f%%',
                   startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            ax.axis()
            plt.title("Sentence Tone Distribution", fontsize=12, pad=20)
            plt.tight_layout()
            chart_path = "pie_chart.png"
            plt.savefig(chart_path, dpi=100)
            plt.close()
        else:
            chart_path = None

        html_output = "<div style='font-family: times-new-roman; line-height: 2.6;'>" + \
                      "".join(formatted_lines) + "</div>"

        return (html_output, chart_path) if chart_path else (html_output, None)

    except:
        return None

interface = gr.Interface(
    fn=gradio_classify,
    inputs=gr.Textbox(label="Input Text", lines=5, placeholder="Paste your text here..."),
    outputs=[
        gr.HTML(label="Analysis Results"),
        gr.Image(label="Tone Distribution", show_label=True)
    ],
    title="📊 Text Tonality Analyzer",
    description="Classifies text and shows tone distribution. Color key: "
                "<span style='color: #00bfff'>Informative</span> | "
                "<span style='color: #DC143C'>Argumentative</span> | "
                "<span style='color: #FFC000'>Persuasive</span>",
    css="""
    .gradio-container {max-width: 1000px !important}
    textarea {font-size: 16px !important}
    .output_image img {max-width: 600px !important; margin-top: 20px !important}
    """
)

interface.launch(share=True)
