import requests
import gutenberg_cleaner

book_ids = [
    5200,
    3813,
    5120,
    6800,
    1105,
    1128,
    1556,
    4363,
    5560,
    2429,
    19942,
    1533,
    5110,
    5296,
    30155,
    1600,
    1717,
    1129,
    347,
    46520,
    5027,
    4981,
    1342,
    84,
    2701,
    98,
    74,
    1661,
    1080,
    4300,
    121,
    514,
    159,
    1400,
    174,
    25344,
    18884,
    2148,
    844,
    120,
    2542,
    2814,
    2160,
    2852,
    30254,
    244,
    4300,
    844,
    345,
    76,
    768,
    1200,
    158,
    1929,
    11,
    16328,
    1200,
    844,
    2554,
    1080,
    1342,
    1661,
    98,
    2701,
    2542,
    11,
    76,
    345,
    236,
    120,
    5200,
    237,
    1232,
    1727,
    8440,
    20203,
    768,
    2638,
    34514,
    6367,
    5500,
    23042,
    996,
    19033,
    203,
    7687,
    2148,
    20417,
    8800,
    35,
    356,
    21100,
    1322,
    205,
    12345,
]


def get_url(book_id):
    return f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


def count_tokens(text):
    # simple tokenizer: split by whitespace
    return len(text.split())


output_text = ""
token_count = 0

for book_id in book_ids:
    url = get_url(book_id)
    print(f"Downloading from {url}")

    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download book {book_id}: {e}")
        continue

    cleaned = gutenberg_cleaner.super_cleaner(r.text)
    cleaned_lines = [
        line
        for line in cleaned.splitlines()
        if line.strip() and "[deleted]" not in line and "[Illustration:" not in line
    ]
    cleaned_filtered = "\n".join(cleaned_lines)

    new_tokens = count_tokens(cleaned_filtered)

    output_text += cleaned_filtered + "\n\n"
    token_count += new_tokens
    print(f"Accumulated tokens: {token_count}")

print(f"Total tokens collected: {token_count}")

with open("dataset.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Saved cleaned dataset to dataset.txt")
