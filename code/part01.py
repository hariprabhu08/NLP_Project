import assemblyai as aai

aai.settings.api_key = "79e18bfc5d8b4c0e9060ecb014fcfd0b"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("/content/drive/MyDrive/Colab_Notebooks/nlptts/pre_output/output_audio.wav")

# Specify the directory path where the output file should be saved
output_directory = "/content/drive/MyDrive/Colab_Notebooks/nlptts/tained_model"

# Write the transcript and timestamps to a file
output_file_path = output_directory + "output.txt"
with open(output_file_path, "w") as output_file:
    output_file.write("Transcript:\n")
    output_file.write(transcript.text + "\n\n")
    output_file.write("Word-level timestamps:\n")
    for word in transcript['words']:
        output_file.write(f"{word['start']} - {word['end']}: {word['text']}\n")

print("Transcript and timestamps saved to:", output_file_path)
