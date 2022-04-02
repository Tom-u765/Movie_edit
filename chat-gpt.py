from transformers import T5Tokenizer, AutoModelForCausalLM

model_name = 'rinna/japanese-gpt2-medium'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import voice_talk
from voice_talk import MicrophoneStream

# ここをjuntalkをモジュールにすれば会話が実現する。
input_message = input("会話を入力：")


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

"""
for i, output_sequence in enumerate(output_sequences):
    print(f"=== 文章生成 {i + 1} ===")
    output_sequence = output_sequence.tolist()

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
"""
output_sequence = output_sequences[0] #適当に一番だけ選んでるので品質はクソ
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
