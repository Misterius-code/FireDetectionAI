import discord
from discord.ext import commands
import os
import uuid
import nest_asyncio
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import asyncio  # DODANE

# === Ustawienia ===
TOKEN = "TWOJ TOKEN"
SAVE_FOLDER = "images"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

# Tworzymy folder na obrazy, jeśli nie istnieje
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Konfiguracja intents
intents = discord.Intents.default()
intents.message_content = True

# Inicjalizacja bota
bot = commands.Bot(command_prefix="!", intents=intents)

# === Funkcja testowania obrazu ===
def test_image(file_path):
    try:
        np.set_printoptions(suppress=True)

        # Załaduj model z obsługą błędu DepthwiseConv2D
        model = load_model(MODEL_PATH, custom_objects={"DepthwiseConv2D": tf.keras.layers.DepthwiseConv2D})

        # Załaduj etykiety klas
        with open(LABELS_PATH, "r") as f:
            class_names = f.readlines()

        # Przetwarzanie obrazu
        image = Image.open(file_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Predykcja
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)

        # Wynik
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        return class_name, confidence_score

    except Exception as e:
        return f"Błąd: {str(e)}", 0

# === Obsługa zdarzenia gotowości ===
@bot.event
async def on_ready():
    print(f'Zalogowano jako {bot.user}')

# === Komenda do przesyłania obrazów ===
@bot.command()
async def upload(ctx):
    if not ctx.message.attachments:
        await ctx.send("Prześlij obraz jako załącznik!")
        return

    for attachment in ctx.message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            unique_filename = f"{uuid.uuid4()}_{attachment.filename}"
            file_path = os.path.join(SAVE_FOLDER, unique_filename)

            await attachment.save(file_path)

            class_name, confidence_score = test_image(file_path)

            await ctx.send(f"**Wykryto:** {class_name}\n**Pewność:** {confidence_score:.2%}")

            if "Ogien" in class_name:
                await ctx.send("🚨 Uwaga! Wykryto ogień! Zadzwoń na straż pożarną!")
            else:
                await ctx.send("✅ Brak ognia.")

        else:
            await ctx.send(f"Plik {attachment.filename} nie jest obsługiwanym formatem obrazu.")

# Uruchomienie bota w Google Colab
nest_asyncio.apply()
loop = asyncio.get_event_loop()
loop.run_until_complete(bot.start(TOKEN))
