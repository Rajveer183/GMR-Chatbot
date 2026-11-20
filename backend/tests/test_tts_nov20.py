"""Play-button speech: text rewritten for the voice, short chunks, joined WAV (no API calls)."""

import struct

from shared import tts


def test_speakable_rewrites_symbols_markup_and_dates():
    s = tts.speakable
    assert s("AZAMGARH / EDC BALLIA / EDD BALLIA NAGAR") == "AZAMGARH, EDC BALLIA, EDD BALLIA NAGAR"
    assert s("**Kashi** up 4.5% &amp; Agra <b>3.2L</b>, ₹1.2 Cr") == \
        "Kashi up 4.5 percent and Agra 3.2 lakh, rupees 1.2 crore"
    assert s("2026-07-01 and 2026-08") == "1 July 2026 and August 2026"
    assert s("delta -6, ratio 3/4") == "delta minus 6, ratio 3/4"
    assert s("3-30 Days, 31–60 days") == "3 to 30 Days, 31 to 60 days"
    assert s("age_60_plus ~ 212,481") == "age 60 plus about 212,481"
    assert s("PF ≥ 0.9 and loss > 20%") == "PF at least 0.9 and loss more than 20 percent"
    # Square brackets are vocal directions to the voice model: never sent.
    assert "[" not in s("see [note] here")
    # Lines and bullets become sentences instead of running together.
    assert s("Top:\n- A: 12\n- B: 9") == "Top: A: 12. B: 9"
    # Codes keep their hyphen.
    assert "4262132-EDSD" in s("sub division 4262132-EDSD SIKANDERPUR")


def test_chunks_are_short_whole_sentences():
    text = " ".join(f"Sentence number {i} has a few words in it." for i in range(30))
    chunks = tts.chunk_text(text, 180)
    assert all(len(c) <= 180 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c.endswith(".") for c in chunks)


def test_overlong_sentence_splits_at_commas_then_words():
    text = ", ".join(f"division EDD {i} BALLIA NAGAR" for i in range(20)) + "."
    chunks = tts.chunk_text(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == text


def test_cap_ends_on_a_sentence():
    text = "One two three. " * 100
    capped = tts._cap(text.strip(), 200)
    assert len(capped) <= 200 and capped.endswith(".")


def _stream_wav(samples: bytes) -> bytes:
    # Groq's shape: streaming placeholders for the sizes plus a LIST chunk.
    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    info = b"INFOISFT\x05\x00\x00\x00Lavf\x00\x00"
    return (b"RIFF\xff\xff\xff\xffWAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"LIST" + struct.pack("<I", len(info)) + info + b"data\xff\xff\xff\xff" + samples)


def test_join_wavs_concatenates_pcm_with_valid_header():
    a, b = b"\x01\x00" * 100, b"\x02\x00" * 50
    out = tts._join_wavs([_stream_wav(a), _stream_wav(b)])
    fmt, pcm = tts._wav_parts(out)
    gap = int(24000 * 0.12) * 2
    assert pcm == a + b"\x00" * gap + b
    assert struct.unpack("<I", out[4:8])[0] == len(out) - 8
