#!/usr/bin/env python3

# for transparency, i had claude generate this.
# if this note is not present in a file, then it is my own work :)

import os


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def is_similar_word(word1, word2, threshold=1):
    w1, w2 = word1.lower(), word2.lower()

    # Only filter exact substring matches for longer words
    if len(w1) > 6 and len(w2) > 6 and (w1 in w2 or w2 in w1):
        return True

    # Only filter exact matches (threshold 0)
    return levenshtein_distance(w1, w2) == 0


def filter_similar_words(input_file, output_file, threshold=0):
    os.makedirs("dataset", exist_ok=True)

    # Use original file from dataset if it exists
    source_file = f"dataset/{os.path.basename(input_file)}"
    if not os.path.exists(source_file):
        os.rename(input_file, source_file)

    with open(source_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # Keep short words, just filter non-alphabetic
    words = [w for w in words if w.isalpha()]

    filtered_words = []
    removed_count = 0

    for word in words:
        # Check similarity with last few added words
        is_duplicate = False
        check_range = min(3, len(filtered_words))  # Only check last 3 words

        for j in range(check_range):
            if is_similar_word(word, filtered_words[-(j+1)], threshold):
                is_duplicate = True
                removed_count += 1
                break

        if not is_duplicate:
            filtered_words.append(word)

    with open(output_file, "w", encoding="utf-8") as f:
        for word in filtered_words:
            f.write(word + "\n")

    print(f"Original: {len(words)} words (after basic filtering)")
    print(f"Filtered: {len(filtered_words)} words")
    print(f"Removed: {removed_count} words")


if __name__ == "__main__":
    # Use the original 1.9M word file directly
    filter_similar_words("data/dataset/wordlist-german.txt", "data/de.txt", threshold=0)
