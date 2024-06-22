from flask import Flask, request, jsonify
import torch
import numpy as np
import ChatTTS
import os
from scipy.io.wavfile import write
from pydub import AudioSegment  # 导入用于处理音频的库
import simpleaudio as sa  # 导入用于播放音频的库

app = Flask(__name__)

# Load ChatTTS model
chat = ChatTTS.Chat()
chat.load_models()

# Function to convert numpy array to AudioSegment
def numpy_to_audiosegment(audio_data, sample_rate):
    audio_data = (audio_data * 32767).astype(np.int16)
    audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=sample_rate, sample_width=audio_data.dtype.itemsize, channels=1)
    return audio_segment

# Function to generate and play audio
def generate_and_play_audio(text, temperature, audio_seed_input):
    try:
        # Set seed if specified
        if audio_seed_input != -1:
            torch.manual_seed(audio_seed_input)

        # Generate audio using ChatTTS model
        params_infer_code = {'spk_emb': chat.sample_random_speaker(), 'temperature': temperature}
        params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
        
        # Print current speaker settings to the terminal
        print(f"当前音色设置: {params_infer_code['spk_emb']}")

        wav = chat.infer([text], params_refine_text=params_refine_text, params_infer_code=params_infer_code, use_decoder=True)
        audio_data = np.array(wav[0]).flatten()
        sample_rate = 24000

        # Convert numpy array to AudioSegment
        audio_segment = numpy_to_audiosegment(audio_data, sample_rate)

        # Play generated audio using simpleaudio
        play_obj = sa.play_buffer(audio_segment.raw_data, num_channels=audio_segment.channels, bytes_per_sample=audio_segment.sample_width, sample_rate=audio_segment.frame_rate)

        # Wait until playback is finished
        play_obj.wait_done()

        return sample_rate, audio_data

    except Exception as e:
        raise RuntimeError(f"Error generating or playing audio: {e}")

# Endpoint for generating audio
@app.route('/generate_audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json()

        # Extract parameters from request
        text = data['text']
        temperature = float(data['temperature'])
        audio_seed_input = int(data.get('audio_seed_input', -1))

        # Generate and play audio
        sample_rate, audio_data = generate_and_play_audio(text, temperature, audio_seed_input)

        # Save audio to a file
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'generated_audio.wav')
        write(output_path, sample_rate, audio_data.astype(np.int16))

        # Delete the audio file after playing
        os.remove(output_path)

        return jsonify({
            'sample_rate': sample_rate,
            'audio_data_length': len(audio_data),
            'output_path': output_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
