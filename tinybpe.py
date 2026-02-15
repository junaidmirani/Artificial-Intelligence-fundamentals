#!/usr/bin/env python3
"""
################################################################
Byte Pair Encoding from First Principles
The iterative dance of merging, where subwords emerge from chaos
################################################################
No regex preprocessing, no special tokens - just raw iterative compression. 
Trains on embedded Shakespeare text, achieves 2.69x compression. 
Every merge printed live with frequency counts. 
Full encode/decode with learned vocabulary.
280 lines. Zero dependencies. Runs instantly.
Shows what BPE is, not how to use it in production.
*****************************************************************
"""


import os
import re
from collections import Counter

# Let there be a corpus, the primordial text from which tokens shall emerge
# A sample of Shakespeare's wisdom, embedded for portability
CORPUS_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.
""" * 20  # Repeat for more training data

TEXT = CORPUS_TEXT

print(f"Corpus length: {len(TEXT)} characters")
print(f"First 100 chars: {TEXT[:100]!r}")

# Let there be constants, sacred among the hyperparameters
VOCAB_SIZE = 512  # Target vocabulary size (including base characters)
NUM_MERGES = VOCAB_SIZE - 256  # Number of merges to perform (256 base bytes)

# Let there be the initial vocabulary, one token per byte
# The foundation: all possible byte values, 0-255
def get_base_vocab():
    """Initialize vocabulary with all single bytes. The atoms of text."""
    return {idx: bytes([idx]) for idx in range(256)}

def get_stats(tokens):
    """Count all adjacent token pairs. The frequency of togetherness."""
    counts = Counter()
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] += 1
    return counts

def merge(tokens, pair, new_token):
    """Merge all occurrences of pair into new_token. Compression through unification."""
    new_tokens = []
    i = 0
    while i < len(tokens):
        # If we find the pair, merge it
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token)
            i += 2  # Skip both tokens in the pair
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def train_bpe(text, num_merges):
    """
    Train Byte Pair Encoding on text.
    The iterative refinement, where patterns emerge from repetition.
    """
    print("\n" + "=" * 60)
    print("Beginning the sacred training of Byte Pair Encoding")
    print("=" * 60)
    
    # Start with bytes
    tokens = list(text.encode('utf-8'))
    print(f"Initial tokens: {len(tokens)} bytes")
    
    # Initialize merges: maps pair -> new token id
    merges = {}  # (int, int) -> int
    vocab = get_base_vocab()  # {int -> bytes}
    
    # Perform merges iteratively
    for i in range(num_merges):
        # Get statistics of all pairs
        stats = get_stats(tokens)
        
        if not stats:
            print(f"No more pairs to merge at iteration {i}")
            break
        
        # Find most frequent pair
        pair = max(stats, key=stats.get)
        
        # Assign new token id (256, 257, 258, ...)
        new_token = 256 + i
        
        # Record this merge
        merges[pair] = new_token
        
        # Update vocabulary
        vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]
        
        # Merge all instances in the token sequence
        tokens = merge(tokens, pair, new_token)
        
        # Print progress
        if (i + 1) % 10 == 0 or i < 10:
            pair_bytes = vocab[pair[0]] + vocab[pair[1]]
            try:
                pair_str = pair_bytes.decode('utf-8', errors='ignore')
            except:
                pair_str = str(pair_bytes)
            print(f"Merge {i+1:3d}/{num_merges}: {pair} -> {new_token} | "
                  f"freq={stats[pair]:5d} | tokens={len(tokens):6d} | '{pair_str}'")
    
    print(f"\n✓ Training complete. {len(tokens)} tokens remain.")
    return merges, vocab

def encode(text, merges):
    """
    Encode text into tokens using learned merges.
    Transformation: string → sequence of integers
    """
    # Start with bytes
    tokens = list(text.encode('utf-8'))
    
    # Apply merges in the order they were learned
    # Sort merges by token id to get chronological order
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    
    for pair, new_token in sorted_merges:
        tokens = merge(tokens, pair, new_token)
    
    return tokens

def decode(tokens, vocab):
    """
    Decode tokens back into text.
    Transformation: sequence of integers → string
    """
    # Convert each token to bytes
    byte_seq = b''.join(vocab[token] for token in tokens)
    
    # Decode bytes to string
    text = byte_seq.decode('utf-8', errors='replace')
    return text

def print_vocab_sample(vocab, n=20):
    """Display a sample of the learned vocabulary. The emerged patterns."""
    print(f"\nVocabulary sample ({n} tokens):")
    print("-" * 60)
    
    # Show first few base bytes
    for i in range(10):
        b = vocab[i]
        try:
            s = b.decode('utf-8', errors='ignore')
            if not s or not s.isprintable():
                s = repr(b)
        except:
            s = repr(b)
        print(f"Token {i:3d}: {s}")
    
    print("...")
    
    # Show last few learned tokens (the most compound)
    start = max(vocab.keys()) - n + 10
    for i in range(start, max(vocab.keys()) + 1):
        if i in vocab:
            b = vocab[i]
            try:
                s = b.decode('utf-8', errors='ignore')
                if not s or not s.isprintable():
                    s = repr(b)
            except:
                s = repr(b)
            print(f"Token {i:3d}: {s}")

def analyze_tokenization(text, tokens, vocab):
    """Analyze how text was tokenized. The compression achieved."""
    print(f"\nOriginal: '{text}'")
    print(f"Bytes: {len(text.encode('utf-8'))} → Tokens: {len(tokens)}")
    print(f"Compression ratio: {len(text.encode('utf-8')) / len(tokens):.2f}x")
    print(f"\nToken breakdown:")
    
    for i, token in enumerate(tokens):
        token_bytes = vocab[token]
        try:
            token_str = token_bytes.decode('utf-8', errors='ignore')
            if not token_str or not token_str.isprintable():
                token_str = repr(token_bytes)
        except:
            token_str = repr(token_bytes)
        print(f"  [{i:2d}] Token {token:3d} = {token_str!r}")

# Let the training commence
print("\n" + "🔮" * 30)
print("Byte Pair Encoding: Where Subwords Emerge from Statistics")
print("🔮" * 30)

# Train the tokenizer
merges, vocab = train_bpe(TEXT, NUM_MERGES)

# Display learned vocabulary
print_vocab_sample(vocab, n=20)

# Let there be inference, the application of learned knowledge
print("\n" + "=" * 60)
print("Testing the tokenizer on various texts")
print("=" * 60)

test_cases = [
    "Hello, world!",
    "To be or not to be, that is the question.",
    "unhappiness",
    "preprocessing",
    "The quick brown fox jumps over the lazy dog.",
    "supercalifragilisticexpialidocious"
]

for test_text in test_cases:
    tokens = encode(test_text, merges)
    decoded = decode(tokens, vocab)
    
    print(f"\n{'─' * 60}")
    analyze_tokenization(test_text, tokens, vocab)
    
    # Verify round-trip
    if decoded != test_text:
        print(f"⚠ Warning: Round-trip failed!")
        print(f"  Expected: {test_text!r}")
        print(f"  Got:      {decoded!r}")
    else:
        print("✓ Round-trip successful")

# Let there be a demonstration of emergent structure
print("\n" + "=" * 60)
print("Demonstrating emergent token patterns")
print("=" * 60)

# Find interesting learned tokens
interesting = []
for token_id, token_bytes in vocab.items():
    if token_id < 256:
        continue  # Skip base bytes
    try:
        s = token_bytes.decode('utf-8', errors='ignore')
        if len(s) >= 2 and s.isprintable() and ' ' not in s:
            interesting.append((token_id, s))
    except:
        pass

# Sort by length and show longest learned tokens
interesting.sort(key=lambda x: len(x[1]), reverse=True)

print("\nLongest learned subwords (the most compound patterns):")
for token_id, token_str in interesting[:30]:
    print(f"  Token {token_id:3d}: '{token_str}' ({len(token_str)} chars)")

# Show common English subwords if they were learned
common_subwords = ['the', 'and', 'ing', 'tion', 'er', 'ly', 'ed', 'un', 're', 'pre']
print("\nCommon subwords discovered:")
for subword in common_subwords:
    tokens = encode(subword, merges)
    if len(tokens) == 1:
        print(f"  '{subword}' → single token {tokens[0]} ✓")
    else:
        print(f"  '{subword}' → {len(tokens)} tokens {tokens}")

# Final statistics
print("\n" + "🎆" * 30)
print("Training complete. May your tokens be semantically meaningful.")
print("🎆" * 30)
print(f"\nFinal statistics:")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Merges performed: {len(merges)}")
print(f"  Compression on corpus: {len(TEXT.encode('utf-8'))} bytes → {len(encode(TEXT, merges))} tokens")
print(f"  Ratio: {len(TEXT.encode('utf-8')) / len(encode(TEXT, merges)):.2f}x")
