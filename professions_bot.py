import telebot
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

url = 'https://raw.githubusercontent.com/Vlad228-cash/profession-recommender-bot/refs/heads/main/professions.json'
response = requests.get(url)
professions = json.loads(response.text)

telegram_bot = 'Telegram_Token'
bot = telebot.TeleBot(telegram_bot)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Привет! Я Profession бот 👔\nОпишите чем вы любите заниматься (мин 10 слов)")

@bot.message_handler(content_types=['text'])
def profession(message):
    if len(message.text.split()) < 10:
        bot.send_message(message.chat.id,"Слишком мало слов")
        return
    candidate = clean_text(message.text)
    profession_vectors, candidate_vectors = build_vectors(
        professions,
        candidate
    )
    recommend_for_candidate(
        candidate_vectors[0],
        profession_vectors,
        message
    )

def clean_text(text):
    text = text.lower()
    symbols = ['.',',','!','?',':',';',')','(','"',"'"]
    for symbol in symbols:
        text = text.replace(symbol,'')
    return text

def build_vectors(professions, candidate):
    all_text = list(professions.values()) + [candidate]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_text)

    profession_count = len(professions)
    profession_vectors = matrix[:profession_count]
    candidate_vectors = matrix[profession_count:]
    return profession_vectors, candidate_vectors

def recommend_for_candidate(candidate_vector,profession_vectors,message):
    similarities = cosine_similarity(candidate_vector,profession_vectors)[0]
    best_index = np.argmax(similarities)
    profession_names = list(professions.keys())
    best_profession = profession_names[best_index]
    best_score = similarities[best_index]
    sorted_indices = np.argsort(similarities)[::-1]

    s = ''
    for place, index in enumerate(sorted_indices[:3], start=1):
        name = profession_names[index]
        score = similarities[index]
        s += f"{place}. {name} - {score*100:.2f}%\n"
    bot.send_message(message.chat.id, f"Рекомендация: {best_profession}\nСходство: {best_score*100}%\n\nТоп направлений:\n{s}")

bot.polling()



