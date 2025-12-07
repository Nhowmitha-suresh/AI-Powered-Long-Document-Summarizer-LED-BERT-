from transformers import LEDTokenizer, LEDForConditionalGeneration

def summarize_with_led(text):
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

    # Encode long text
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=16000
    )

    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=200,
        min_length=50,
        length_penalty=2.0
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    with open("huge_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    summary = summarize_with_led(text)
    print("\nSUMMARY:\n", summary)
