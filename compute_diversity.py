#!/usr/bin/env python
import argparse
import json
import random
from glob import glob
import torch
from sentence_transformers import SentenceTransformer
def extract_text(obj, field_name: str):
    """
    Extract a text string from a JSON object.
    field_name:
      - "auto": try several common fields.
      - any other string: try obj[field_name] as string first, then fall back to auto heuristics.
    """
    # 1) If user explicitly gave a field, try that first
    if field_name != "auto":
        if field_name in obj and isinstance(obj[field_name], str):
            txt = obj[field_name].strip()
            if txt:
                return txt
    # 2) Automatic heuristics
    for key in ["output", "response", "text", "completion", "response_text"]:
        if key in obj and isinstance(obj[key], str):
            txt = obj[key].strip()
            if txt:
                return txt
    # 3) List-of-dicts style outputs: e.g. "outputs": [{"text": "..."}]
    for key in ["outputs", "samples", "generations", "responses"]:
        if key in obj and isinstance(obj[key], list) and obj[key]:
            first = obj[key][0]
            if isinstance(first, dict):
                for tk in ["text", "response", "output", "completion"]:
                    if tk in first and isinstance(first[tk], str):
                        txt = first[tk].strip()
                        if txt:
                            return txt
    # 4) Nested dict in "response" or "output"
    for key in ["response", "output"]:
        if key in obj and isinstance(obj[key], dict):
            inner = obj[key]
            for tk in ["text", "response", "completion"]:
                if tk in inner and isinstance(inner[tk], str):
                    txt = inner[tk].strip()
                    if txt:
                        return txt
    return None
def load_texts_sampled(files, field_name: str, max_texts: int, log_every: int = 10000):
    """
    Stream all JSONL files and keep at most `max_texts` texts using reservoir sampling.
    This avoids loading everything into memory and keeps complexity manageable.
    """
    texts = []
    seen = 0
    n_lines = 0
    n_objs = 0
    for pattern in files:
        paths = sorted(glob(pattern))
        print(f"Pattern '{pattern}' matched {len(paths)} files.")
        for path in paths:
            print(f"  Reading: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    n_lines += 1
                    if n_lines % log_every == 0:
                        print(
                            f"    Processed {n_lines} lines, "
                            f"{seen} texts seen, {len(texts)} kept in reservoir."
                        )
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n_objs += 1
                    txt = extract_text(obj, field_name)
                    if not txt:
                        continue
                    # Reservoir sampling
                    seen += 1
                    if len(texts) < max_texts:
                        texts.append(txt)
                    else:
                        j = random.randint(0, seen - 1)
                        if j < max_texts:
                            texts[j] = txt
    print(
        f"\nFinished reading.\n"
        f"  Total lines read: {n_lines}\n"
        f"  JSON objects parsed: {n_objs}\n"
        f"  Texts seen (with non-empty extracted text): {seen}\n"
        f"  Texts kept in reservoir: {len(texts)}"
    )
    return texts
def compute_avg_cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    embeddings: (N, D) tensor, assumed *not* normalized.
    Returns average cosine similarity over all unordered pairs i<j.
    """
    emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    n = emb.size(0)
    if n < 2:
        return float("nan")
    # Cosine similarity matrix
    cos_mat = emb @ emb.T  # (N, N)
    total_sum = cos_mat.sum()
    diag_sum = torch.diagonal(cos_mat).sum()
    pair_sum = (total_sum - diag_sum) / 2.0  # each pair appears twice
    num_pairs = n * (n - 1) / 2.0
    avg_sim = pair_sum / num_pairs
    return avg_sim.item()
def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity from rollout JSONL files using BGE-M3 (with sampling)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSONL files or globs (e.g. 'verl/rollout_outputs/*.jsonl').",
    )
    parser.add_argument(
        "--field",
        default="auto",
        help="Field name to read from each JSON object. "
             "Use 'auto' to try common names like 'output', 'response', etc. Default: auto",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="SentenceTransformer model name or path. Default: BAAI/bge-m3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding. Default: 32",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g. 'cuda', 'cuda:0', 'cpu'). "
             "Default: auto-detect cuda if available else cpu.",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=2000,
        help="Maximum number of texts to keep via reservoir sampling. "
             "Controls complexity (O(max_texts^2)). Default: 2000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reservoir sampling. Default: 0",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    # Detect device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading texts from: {args.inputs}")
    texts = load_texts_sampled(args.inputs, args.field, max_texts=args.max_texts, log_every=10000)
    print(f"\nLoaded {len(texts)} texts for embedding (sampled).")
    if len(texts) < 2:
        print("Need at least 2 texts to compute pairwise similarity.")
        return
    print(f"\nLoading model: {args.model_name} on device: {device}")
    model = SentenceTransformer(args.model_name, device=device)
    print("Encoding texts...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
    )
    print("Computing pairwise cosine similarities...")
    avg_sim = compute_avg_cosine_similarity(embeddings)
    diversity = 1.0 - avg_sim
    print("\n=== Results ===")
    print(f"# sampled responses: {len(texts)}")
    print(f"Average pairwise cosine similarity: {avg_sim:.6f}")
    print(f"Diversity (1 - avg_cosine_similarity): {diversity:.6f}")
if __name__ == "__main__":
    main()