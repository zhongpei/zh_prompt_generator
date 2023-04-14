import random
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoProcessor
from transformers import pipeline, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
big_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
big_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

text_pipe = pipeline('text-generation', model='succinctly/text2image-prompt-generator')

zh2en_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-zh-en').eval()
zh2en_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").eval()
en2zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")


def load_prompter():
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return prompter_model, tokenizer


prompter_model, prompter_tokenizer = load_prompter()


def generate_prompter(plain_text, max_new_tokens=75, num_beams=8, num_return_sequences=8, length_penalty=-1.0):
    input_ids = prompter_tokenizer(plain_text.strip() + " Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        length_penalty=length_penalty
    )
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result = []
    for output_text in output_texts:
        result.append(output_text.replace(plain_text + " Rephrase:", "").strip())

    return "\n".join(result)


def translate_zh2en(text):
    with torch.no_grad():
        encoded = zh2en_tokenizer([text], return_tensors='pt')
        sequences = zh2en_model.generate(**encoded)
        return zh2en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


def translate_en2zh(text):
    with torch.no_grad():
        encoded = en2zh_tokenizer([text], return_tensors="pt")
        sequences = en2zh_model.generate(**encoded)
        return en2zh_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]


def text_generate(text_in_english):
    seed = random.randint(100, 1000000)
    set_seed(seed)

    result = ""
    for _ in range(6):
        sequences = text_pipe(text_in_english, max_length=random.randint(60, 90), num_return_sequences=8)
        list = []
        for sequence in sequences:
            line = sequence['generated_text'].strip()
            if line != text_in_english and len(line) > (len(text_in_english) + 4) and line.endswith(
                    (':', '-', '—')) is False:
                list.append(line)

        result = "\n".join(list)
        result = re.sub('[^ ]+\.[^ ]+', '', result)
        result = result.replace('<', '').replace('>', '')
        if result != '':
            break
    return result, "\n".join(translate_en2zh(line) for line in result.split("\n") if len(line) > 0)


def get_prompt_from_image(input_image):
    image = input_image.convert('RGB')
    pixel_values = big_processor(images=image, return_tensors="pt").to(device).pixel_values

    generated_ids = big_model.to(device).generate(pixel_values=pixel_values, max_length=50)
    generated_caption = big_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    return generated_caption


with gr.Blocks() as block:
    with gr.Column():
        with gr.Tab('文本生成'):
            with gr.Row():
                input_text = gr.Textbox(lines=6, label='你的想法', placeholder='在此输入内容...')
                translate_output = gr.Textbox(lines=6, label='翻译结果(Prompt输入)')

            with gr.Accordion('SD优化参数设置', open=False):
                max_new_tokens = gr.Slider(1, 255, 75, label='max_new_tokens', step=1)
                nub_beams = gr.Slider(1, 30, 8, label='num_beams', step=1)
                num_return_sequences = gr.Slider(1, 30, 8, label='num_return_sequences', step=1)
                length_penalty = gr.Slider(-1.0, 1.0, -1.0, label='length_penalty')

            generate_prompter_output = gr.Textbox(lines=6, label='SD优化的 Prompt')

            output = gr.Textbox(lines=6, label='瞎编的 Prompt')
            output_zh = gr.Textbox(lines=6, label='瞎编的 Prompt(zh)')
            with gr.Row():
                translate_btn = gr.Button('翻译')
                generate_prompter_btn = gr.Button('SD优化')
                gpt_btn = gr.Button('瞎编')

        with gr.Tab('从图片中生成'):
            with gr.Row():
                input_image = gr.Image(type='pil')
            img_btn = gr.Button('提交')
            output_image = gr.Textbox(lines=6, label='生成的 Prompt')
    translate_btn.click(
        fn=translate_zh2en,
        inputs=input_text,
        outputs=translate_output
    )
    generate_prompter_btn.click(
        fn=generate_prompter,
        inputs=[translate_output, max_new_tokens, nub_beams, num_return_sequences, length_penalty],
        outputs=generate_prompter_output
    )
    gpt_btn.click(
        fn=text_generate,
        inputs=translate_output,
        outputs=[output, output_zh]
    )
    img_btn.click(
        fn=get_prompt_from_image,
        inputs=input_image,
        outputs=output_image
    )

block.queue(max_size=64).launch(show_api=False, enable_queue=True, debug=True, share=False, server_name='0.0.0.0')
