

import tensorflow as tf
from analyzer import Analyzer

# 1. Initiera
analysis_tool = Analyzer()

# 2. Skapa/Ladda modell (vi använder en enkel placeholder-modell här)
test_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7, activation='softmax')
])

# 3. KÖR ANALYSEN
# Ersätt 'video.mkv' med namnet på din fil
File_path = 'video.mkv' 

print(f"Startar analys av {File_path}...")
success, results = analysis_tool.analyze(model=test_model, file=File_path)

if success:
    print("Analysen är klar!")
    print("Här är de första raderna av resultatet:")
    print(results.head()) # Visar tabellen med bildrutor och känslor