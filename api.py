from flask import Flask, request, jsonify
import torch
import numpy as np
import ChatTTS
import os
from scipy.io.wavfile import write

app = Flask(__name__)

# Load ChatTTS model
chat = ChatTTS.Chat()
chat.load_models()

# Endpoint for generating audio
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()

        # Extract parameters from request
        text = data['text']
        temperature = float(data['temperature'])
        audio_seed_input = int(data.get('audio_seed_input', -1))

        # Set seed if specified
        if audio_seed_input != -1:
            torch.manual_seed(audio_seed_input)

        # Generate audio using ChatTTS model
        params_infer_code = {'spk_emb': chat.sample_random_speaker(), 'temperature': temperature}
        params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
        wav = chat.infer([text], params_refine_text=params_refine_text, params_infer_code=params_infer_code, use_decoder=True)
        audio_data = np.array(wav[0]).flatten()
        sample_rate = 24000

        # Save audio to a file
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'generated_audio.wav')
        write(output_path, sample_rate, (audio_data * 32767).astype(np.int16))

        return jsonify({
            'sample_rate': sample_rate,
            'audio_data_length': len(audio_data),
            'output_path': output_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
