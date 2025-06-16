import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import struct

def load_glove_vectors(file_path, expected_dim=100):
    """Load GloVe vectors with strict dimension checking"""
    words = []
    vectors = []
    bad_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != expected_dim + 1:  # word + vector
                bad_lines += 1
                continue
                
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == expected_dim:
                    words.append(word)
                    vectors.append(vector)
                else:
                    bad_lines += 1
            except ValueError:
                bad_lines += 1
    
    if bad_lines > 0:
        print(f"Warning: Skipped {bad_lines} malformed lines")
    
    if not vectors:
        raise ValueError("No valid vectors found - check file format")
    
    return words, np.array(vectors)

def compress_glove(input_file, output_file, target_dim=16, quant_bits=4):
    print("Loading and validating vectors...")
    try:
        words, vectors = load_glove_vectors(input_file)
    except Exception as e:
        print(f"Error loading vectors: {e}")
        print("Ensure you're using the correct GloVe file format:")
        print("Each line should contain: word followed by 100 space-separated numbers")
        return

    original_size = vectors.nbytes / (1024 * 1024)
    print(f"Loaded {len(words)} vectors, original size: {original_size:.2f} MB")

    # Can't reduce to higher dimension than original
    target_dim = min(target_dim, vectors.shape[1])
    
    print(f"Reducing from {vectors.shape[1]}D to {target_dim}D...")
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(vectors)
    
    print(f"Quantizing to {quant_bits} bits...")
    max_val = (1 << quant_bits) - 1
    scaler = MinMaxScaler(feature_range=(0, max_val))
    scaled = scaler.fit_transform(reduced)
    quantized = np.round(scaled).astype(np.uint8)
    
    print("Saving compressed vectors...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, vec in zip(words, quantized):
            f.write(f"{word} {' '.join(map(str, vec))}\n")
    
    compressed_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {original_size/compressed_size:.1f}x")

if __name__ == "__main__":
    input_file = "glove.6B.100d.txt"  # Original GloVe file
    output_file = "glove_compressed.txt"
    
    # Conservative settings that should work with any valid GloVe file
    compress_glove(input_file, output_file, target_dim=16, quant_bits=4)
