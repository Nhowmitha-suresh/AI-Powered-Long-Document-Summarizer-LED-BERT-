# examples/summarize.py
from summarizer import Summarizer
import argparse

def run():
    parser = argparse.ArgumentParser(description='Process and summarize lectures')
    parser.add_argument('-path', dest='path', required=True, help='File path of lecture')
    parser.add_argument('-model', dest='model', default='distilbert-base-uncased',
                        help='HuggingFace model name (default: distilbert-base-uncased)')
    parser.add_argument('-hidden', dest='hidden', default='-2',
                        help='Which hidden layer to use from BERT (e.g. -1 or -2). Can also be comma-separated list like "-1,-2"')
    parser.add_argument('-reduce-option', dest='reduce_option', default='mean',
                        help='How to reduce the hidden layer from bert (mean/median/max)')
    parser.add_argument('-greedyness', dest='greedyness', default=0.45,
                        help='Greedyness of the NeuralCoref model (unused if neuralcoref not installed)')

    args = parser.parse_args()

    # Read file with utf-8-sig to automatically strip BOM if present
    with open(args.path, "r", encoding="utf-8-sig") as d:
        text_data = d.read()

    # parse hidden into int or list of ints
    hidden_arg = args.hidden
    if isinstance(hidden_arg, str) and ',' in hidden_arg:
        hidden = [int(x.strip()) for x in hidden_arg.split(',')]
    else:
        try:
            hidden = int(hidden_arg)
        except Exception:
            # fallback: leave as-is (some versions accept list/tuple)
            hidden = hidden_arg

    # coerce greedyness to float
    try:
        greedyness = float(args.greedyness)
    except Exception:
        greedyness = 0.45

    model = Summarizer(
        model=args.model,
        hidden=hidden,
        reduce_option=args.reduce_option
        # If you want to pass a sentence_handler for coref, create and pass it here.
    )

    summary = model(text_data)

    # Normalise returned type and print cleanly
    if isinstance(summary, list):
        out = " ".join(summary)
    else:
        out = summary

    print("\n----- SUMMARY -----\n")
    print(out.strip())
    print("\n-------------------\n")

if __name__ == '__main__':
    run()
