import pandas as pd
import emoji
import re
import nltk

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pathlib import Path
import pandas as pd

# =========================================================
# LOAD SLANG DICTIONARY
# =========================================================



# Lokasi folder dashboard
BASE_DIR = Path(__file__).resolve().parent

# Lokasi file slang_word.csv
SLANG_PATH = BASE_DIR / "data" / "slang_word.csv"

slang_df = pd.read_csv(SLANG_PATH)

slang_dict = dict(
    zip(
        slang_df["slang"],
        slang_df["formal"]
    )
)


# =========================================================
# NORMALIZATION DICTIONARY
# =========================================================

normalization_dict = {

    "amp": "dan",
    "br": " ",
    "yg": "yang",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "jg": "juga",
    "jd": "jadi",
    "tp": "tapi",
    "dr": "dari",
    "dg": "dengan",
    "dgn": "dengan",
    "krn": "karena",
    "utk": "untuk",
    "sdh": "sudah",
    "hrs": "harus",
    "jgn": "jangan",
    "skrg": "sekarang",
    "klo": "kalau",
    "kalo": "kalau",
    "aja": "saja",
    "pake": "pakai",
    "tau": "tahu",
    "akn": "akan",
    "dl": "dulu",
    "brp": "berapa",
    "lg": "lagi",
    "udh": "sudah",
    "bgt": "banget",
    "bngt": "banget",
    "dpet": "dapat",
    "lapker": "lapangan kerja",
    "pk": "pak",
}

slang_dict.update(
    normalization_dict
)


# =========================================================
# STOPWORDS
# =========================================================

nltk.download(
    "stopwords",
    quiet=True
)

stop_words = set(
    stopwords.words("indonesian")
)

# Kata negasi dipertahankan
negation_words = {
    "tidak",
    "bukan",
    "jangan",
    "belum"
}

stop_words = stop_words - negation_words


# =========================================================
# STEMMER
# =========================================================

factory = StemmerFactory()

stemmer = factory.create_stemmer()


# =========================================================
# 1. CASE FOLDING
# =========================================================

def case_folding(text):

    if isinstance(text, str):
        return text.lower()

    return text


# =========================================================
# 2. REMOVE EMOJI & URL
# =========================================================

def remove_emoji_url(text):

    text = str(text)

    # Hapus URL
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Hapus emoji
    text = emoji.replace_emoji(
        text,
        replace=""
    )

    return text


# =========================================================
# 3. REMOVE MENTION & HASHTAG
# =========================================================

def remove_mention_selective(text):

    text = str(text)

    # Pertahankan Jokowi
    # dengan menghilangkan simbol @
    text = re.sub(
        r"@jokowi\b",
        "jokowi",
        text,
        flags=re.IGNORECASE
    )

    # Pertahankan Prabowo
    text = re.sub(
        r"@prabowo\b",
        "prabowo",
        text,
        flags=re.IGNORECASE
    )

    # Hapus mention lainnya
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    return text


# =========================================================
# 4. REMOVE NUMBERS, PUNCTUATION,
#    SPECIAL CHARACTERS & WHITESPACE
# =========================================================

def clean_text(text):

    text = str(text)

    # Hapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Hapus tanda baca dan karakter khusus
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Hapus underscore
    text = re.sub(
        r"_",
        "",
        text
    )

    # Hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Hapus spasi awal dan akhir
    text = text.strip()

    return text


# =========================================================
# 5. NORMALIZATION
# =========================================================

def normalize_text(text):

    words = str(text).split()

    normalized_words = [
        slang_dict.get(
            word.lower(),
            word
        )
        for word in words
    ]

    return " ".join(
        normalized_words
    )


# =========================================================
# 6. REMOVE STOPWORDS
# =========================================================

def remove_stopwords(text):

    words = str(text).split()

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(
        filtered_words
    )


# =========================================================
# 7. STEMMING
# =========================================================

def stemming(text):

    words = str(text).split()

    stemmed_words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(
        stemmed_words
    )


# =========================================================
# MAIN PREPROCESSING FUNCTION
# =========================================================

def preprocess_text(text):

    text = case_folding(text)

    text = remove_emoji_url(text)

    text = remove_mention_selective(text)

    text = clean_text(text)

    text = normalize_text(text)

    text = remove_stopwords(text)

    text = stemming(text)

    return text