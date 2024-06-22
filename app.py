import os
import random
import argparse

import torch
import gradio as gr
import numpy as np

import ChatTTS

print("loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load_models()


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
        }


def deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_audio(text, temperature,audio_seed_input):

    

    if audio_seed_input != -1:
        torch.manual_seed(audio_seed_input)

    # rand_spk = torch.randn(768)

    params_infer_code = {'spk_emb' : chat.sample_random_speaker(), 'temperature': temperature}
    params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}

    wav = chat.infer([text],params_refine_text=params_refine_text, params_infer_code=params_infer_code,use_decoder=True)

    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), text_data]


def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS Webui")
        gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")

        default_text = "菜可能会味道不足，太单调。如果你加得太多，菜可能会味道过重，难以预测它会有多咸。所以，适量的调味通常能做出既不单调也不过分的菜。在语音合成中，温度帮助控制生成的多样性和预测性：低温度让语音听起来更加自然和一致，而高温度让语音更加多样和不可预测。"        
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)

        with gr.Row():
            temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature,越大越发散，越小越保守")
            # top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P")
            # top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K")

        with gr.Row():
            audio_seed_input = gr.Number(value=-1, label="声音种子,-1随机，1男声,2青年女声,3高音男声，4公鸭嗓女声")
            generate_audio_seed = gr.Button("\U0001F3B2")


        generate_button = gr.Button("Generate")

        text_output = gr.Textbox(label="Output Text", interactive=False)
        audio_output = gr.Audio(label="Output Audio")

        generate_audio_seed.click(generate_seed, 
                                  inputs=[], 
                                  outputs=audio_seed_input)


        generate_button.click(generate_audio, 
                              inputs=[text_input, temperature_slider,audio_seed_input], 
                              outputs=[audio_output, text_output])

    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=8080, help='Server port')
    args = parser.parse_args()

    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    main()