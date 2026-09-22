from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))