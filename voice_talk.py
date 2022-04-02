# https://zenn.dev/lovegorira/articles/7f9d0060860a43

from __future__ import division

class Get_log():
    def __init__(self):
        import logging

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger(__name__)

        # ロガーを取得する
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # 出力レベルを設定

        # ハンドラー1を生成する
        h1 = logging.StreamHandler()
        h1.setLevel(logging.DEBUG)  # 出力レベルを設定

        # ハンドラー2を生成する
        h2 = logging.FileHandler('sample.log')
        h2.setLevel(logging.ERROR)  # 出力レベルを設定

        # フォーマッタを生成する
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # ハンドラーにフォーマッターを設定する
        h1.setFormatter(fmt)
        h2.setFormatter(fmt)

        # ロガーにハンドラーを設定する
        logger.addHandler(h1)
        logger.addHandler(h2)

        # ログ出力を行う
        logger.debug("degubログを出力")
        logger.info("infoログを出力")
        logger.warning("warnログを出力")
        logger.error("errorログを出力")
######################################################################################


import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/hoteison/Desktop/Juntalk/Zoom_Jun_Key.json'

######################GPT2関係
from transformers import T5Tokenizer, AutoModelForCausalLM
model_name = 'rinna/japanese-gpt2-medium'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
######################


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer)

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:

            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        overwrite_chars = " " * (num_chars_printed - len(transcript))


        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            # print(type(transcript + overwrite_chars))
            text = transcript + overwrite_chars
            print(f'transcript:{transcript + overwrite_chars}')
            num_chars_printed = 0

##################ここからGPT2
            input_message = transcript + overwrite_chars
            print(input_message)
            input_ids = tokenizer.encode(
                "私: " + input_message + "\nAI: ",
                return_tensors="pt"
            )

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=60,
                temperature=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=10)

            output_sequence = output_sequences[0]  # 適当に一番だけ選んでるので品質はクソ
            # トークンからテキストにデコード
            text = tokenizer.decode(
                output_sequence, clean_up_tokenization_spaces=True)

            total_text = (
                text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
            )

            slice_potsition = total_text.find('AI:')
            edited_text = total_text[:slice_potsition]
            slice_potsition = edited_text.find('私:')
            edited_text = edited_text[:slice_potsition]
            slice_potsition = edited_text.find('<unk>')
            edited_text = edited_text[:slice_potsition]
            slice_potsition = edited_text.find('。')
            edited_text = edited_text[:slice_potsition]
            slice_potsition = edited_text.find('!')
            edited_text = edited_text[:slice_potsition]
            slice_potsition = edited_text.find('?')
            edited_text = edited_text[:slice_potsition]
            print(edited_text)
######################
######################音声生成
            from create_voice import Message
            class Message2(Message):
                pass
            jjj = Message2()
            jjj.inText = edited_text
            jjj.message()
######################




def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "ja-JP"  # a BCP-47 language tag
    print(0)
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code
    )
    print(1)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    print(2)
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        print(3)
        responses = client.streaming_recognize(streaming_config, requests)
        print(4)
        # Now, put the transcription responses to use.
        listen_print_loop(responses)


if __name__ == "__main__":
    print("--main--")
    Get_log()
    main()