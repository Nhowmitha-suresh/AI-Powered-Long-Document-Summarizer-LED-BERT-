# led_hierarchical.py
import argparse
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch
from math import ceil
import os
from tqdm import tqdm

def chunk_by_tokens(tokenizer, text, max_tokens=4000, stride=0):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    L = len(tokens)
    while start < L:
        end = min(start + max_tokens, L)
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if end == L:
            break
        start = end - stride  # overlap if stride>0
    return chunks

def summarize_chunks(model, tokenizer, chunks, device='cpu', batch_size=1, gen_kwargs=None):
    gen_kwargs = gen_kwargs or {}
    summaries = []
    model.to(device)
    model.eval()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding='longest', max_length=tokenizer.model_max_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        for seq in out:
            summaries.append(tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return summaries

def hierarchical_summarize(text, model_name="allenai/led-base-16384", device='cpu',
                           chunk_tokens=5000, chunk_stride=200, batch_size=1,
                           chunk_gen_kwargs=None, final_gen_kwargs=None):
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    # 1) chunk by tokens
    chunks = chunk_by_tokens(tokenizer, text, max_tokens=chunk_tokens, stride=chunk_stride)
    # 2) summarize chunks
    if not chunk_gen_kwargs:
        chunk_gen_kwargs = {
            "num_beams": 2,
            "max_length": 200,
            "min_length": 40,
            "length_penalty": 2.0,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
    chunk_summaries = summarize_chunks(model, tokenizer, chunks, device=device, batch_size=batch_size, gen_kwargs=chunk_gen_kwargs)
    # 3) combine chunk summaries
    combined = "\n".join(chunk_summaries)
    # 4) final summarize
    if not final_gen_kwargs:
        final_gen_kwargs = {
            "num_beams": 4,
            "max_length": 300,
            "min_length": 80,
            "length_penalty": 2.0,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
    final_summary = summarize_chunks(model, tokenizer, [combined], device=device, batch_size=1, gen_kwargs=final_gen_kwargs)
    return {
        "chunks_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "final_summary": final_summary[0] if final_summary else ""
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input text file (utf-8)")
    p.add_argument("--output", "-o", required=False, help="Output file to save final summary")
    p.add_argument("--model", default="allenai/led-base-16384", help="LED model name")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--chunk_tokens", type=int, default=6000, help="Max tokens per chunk (<=16000 recommended)")
    p.add_argument("--chunk_stride", type=int, default=200, help="Overlap tokens between chunks")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for chunk summarization")
    p.add_argument("--final_max_len", type=int, default=300, help="Max tokens for final summary")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # safety: if huge single doc, split more conservatively
    res = hierarchical_summarize(
        text,
        model_name=args.model,
        device=args.device,
        chunk_tokens=args.chunk_tokens,
        chunk_stride=args.chunk_stride,
        batch_size=args.batch_size,
        chunk_gen_kwargs={
            "num_beams": 2, "max_length": 200, "min_length": 40, "length_penalty": 2.0, "early_stopping": True
        },
        final_gen_kwargs={
            "num_beams": 4, "max_length": args.final_max_len, "min_length": 80, "length_penalty": 2.0, "early_stopping": True
        }
    )

    print(f"Chunks: {res['chunks_count']}")
    print("\nFINAL SUMMARY:\n")
    print(res['final_summary'])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(res['final_summary'])
        print(f"\nSaved final summary to {args.output}")

if __name__ == "__main__":
    main()
