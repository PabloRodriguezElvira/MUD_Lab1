import re

# Match any CJK-family character:
# Chinese ideographs and Japanese Hiragana/Katakana.
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF]")

def _has_cjk(text: str) -> bool:
    # True if at least one CJK character is present in the text.
    return bool(_CJK_RE.search(text))

def preprocess(sentence, labels):
    # Normalize each input sentence into a tokenized string.
    processed = []
    for s in sentence:
        # Ensure every element is a valid string.
        s = "" if s is None else str(s)

        if _has_cjk(s):
            # For CJK text, remove spaces and build character bigrams.
            s = re.sub(r"\s+", "", s)
            if len(s) >= 2:
                toks = [s[i:i+2] for i in range(len(s)-1)]
            else:
                # Edge case: single-character CJK string.
                toks = [s]
            processed.append(" ".join(toks))
        else:
            # For non-CJK text, keep only alphanumeric word tokens.
            toks = re.findall(r"\w+", s, flags=re.UNICODE)
            processed.append(" ".join(toks))

    # Return processed texts with the original labels unchanged.
    return processed, labels
