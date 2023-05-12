"""Application that summarizes a Youtube video."""

import os
# import time
import shutil
import librosa
import openai
import soundfile as sf
import youtube_dl

from youtube_dl.utils import DownloadError
from jlog import logger

openai.api_key = "<YOUR_KEY_HERE>"
youtube_url = "<VIDEO_URL>"
outputs_dir = "outputs/"


def find_audio_files(path, extension=".mp3"):
    """Recursively find all audio files in the current directory."""
    audio_files = list()

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                logger.info(f"Found audio file: {file}")
                audio_files.append(os.path.join(root, file))

    return audio_files


def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download audio from a youtube video, and converts it to mp3."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }
        ],
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Downloading audio from Youtube: {youtube_url}")

    # Download fails on first try. This is a hack to force a retry.
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            logger.info("Download complete.")
    except DownloadError as e:
        logger.error(f"Error downloading Youtube video: {e}. Retrying...")
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            logger.info("Download complete.")

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename


def chunk_audio(filename, segment_length: int, output_dir):
    """Segment audio into chunks of a given length."""
    logger.info(f"Segmenting audio file: {filename}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    audio, sr = librosa.load(filename, sr=44100)
    duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(duration / segment_length) + 1

    logger.info(f"Chunking audio into {num_segments} segments.")

    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_files = find_audio_files(output_dir)

    return sorted(chunked_files)


def transcribe_audio(
        audio_files: list, output_file=None, model="whisper-1"
) -> list:
    """Transcribe audio into text."""
    logger.info(f"Transcribing audio files: {audio_files}")

    # logger.info("Throttling API calls. Thread sleep for 1 minute")
    # time.sleep(60)
    transcripts = list()
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        response = openai.Audio.transcribe(model, audio)
        transcripts.append(response["text"])

        # Throttle requests to avoid rate limiting. Free tier BS
        # logger.info("Throttling API calls. Thread sleep for 20 seconds")
        # time.sleep(20)

    if output_file is not None:
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript)

    return transcripts


def summarize(
        chunks: list[str], system_prompt: str,
        model="gpt-3.5-turbo", output_file=None
):
    """Summarize a list of text chunks."""
    logger.info(f"Summarizing audio chunks: {chunks}")

    summaries = list()
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ]
        )

        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)

        # Throttle requests to avoid rate limiting. Free tier BS
        # logger.info("Throttling API calls. Thread sleep for 20 seconds")
        # time.sleep(20)

    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")

    return summaries


def summarize_youtube_video(youtube_url, outputs_dir):
    """Summarize a youtube video."""
    logger.info(f"Summarizing Youtube video: {youtube_url}")

    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks/"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # Max 10 minutes

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )

    transcriptions = transcribe_audio(
        chunked_audio_files, output_file=transcripts_file
    )

    logger.info("Enter a prompt to summarize the video.")
    system_prompt = input()

    summaries = summarize(
        transcriptions, system_prompt=system_prompt, output_file=summary_file
    )

    return "\n".join(summaries)


summary = summarize_youtube_video(youtube_url, outputs_dir)

logger.info(f"\n____SUMMARY____\n{summary}")
