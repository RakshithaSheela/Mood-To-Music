
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def get_emotion(text: str) -> str:
    result = emotion_classifier(text)[0]
    sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
    return sorted_result[0]['label']


# -------------------------
# Spotify Setup
# -------------------------
client_id = "13aa854740b74fe7b58201ac5798f507"
client_secret = "ee446b5b0e184aeda4cdb1714256fb0d"

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(client_id=client_id,
                                          client_secret=client_secret)
)

emotion_to_genre = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "love": "romantic",
    "fear": "chill",
    "surprise": "exciting"
}

def recommend_songs(emotion: str):
    query = "tamil " + emotion_to_genre.get(emotion.lower(), "mood")

    try:
        results = sp.search(q=query, type="track", limit=5)
        tracks = []
        for idx, track in enumerate(results['tracks']['items']):
            name = track['name']
            artist = track['artists'][0]['name']
            url = track['external_urls']['spotify']
            tracks.append(f"{idx+1}. {name} by {artist} - {url}")
        return tracks

    except Exception as e:
        print("⚠️ Error connecting to Spotify:", e)
        return ["Sorry! I couldn't fetch songs right now. Please try again later."]


# -------------------------
# Poem Generator
# -------------------------
generator = pipeline("text-generation", model="gpt2")

def generate_poem(emotion: str) -> str:
    prompt = f"Write a short poem about feeling {emotion}:"
    poem = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return poem


# -------------------------
# Chatbot Function
# -------------------------
def mood_to_music_chatbot(user_text: str):
    emotion = get_emotion(user_text)
    print(f"\nDetected Emotion: {emotion}")

    print("\n🎵 Here are some song recommendations:")
    for song in recommend_songs(emotion):
        print(song)

    print("\n📝 Here's a poem to match your mood:")
    print(generate_poem(emotion))


# -------------------------
# Main Program
# -------------------------
if __name__ == "__main__":
    user_input = input("How are you feeling today?\n")
    mood_to_music_chatbot(user_input)
