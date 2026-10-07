"""Convert predicted token sequences to MusicXML and MIDI (no extra dependencies)."""
from __future__ import annotations

import struct
from xml.sax.saxutils import escape

from .vocab import ALTERS, DURATIONS, KEYS, parse_pitch

DIVISIONS = 4  # per quarter note, enough for sixteenths and dotted eighths
XML_TYPE = {"whole": "whole", "half": "half", "quarter": "quarter",
            "eighth": "eighth", "sixteenth": "16th"}
LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_number(pitch: str) -> int:
    letter, acc, octave = parse_pitch(pitch)
    return 12 * (octave + 1) + LETTER_SEMITONE[letter] + ALTERS[acc]


def _events(tokens: list[str]):
    """Yield (kind, payload) with measures split at barlines."""
    measure: list[tuple[str, str, str]] = []
    attrs: dict[str, str] = {}
    for t in tokens:
        kind, _, val = t.partition("-")
        if kind in ("clef", "keySignature", "timeSignature"):
            attrs[kind] = val
        elif kind == "note":
            p, d = val.split("_")
            measure.append(("note", p, d))
        elif kind == "rest":
            measure.append(("rest", "", val))
        elif t == "barline" and measure:
            yield attrs, measure
            attrs, measure = {}, []
    if measure:
        yield attrs, measure


def to_musicxml(tokens: list[str], title: str = "OMR output") -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<score-partwise version="3.1">',
           f"<work><work-title>{escape(title)}</work-title></work>",
           '<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>',
           '<part id="P1">']
    for n, (attrs, measure) in enumerate(_events(tokens), 1):
        out.append(f'<measure number="{n}">')
        if attrs or n == 1:
            out.append(f"<attributes><divisions>{DIVISIONS}</divisions>")
            if "keySignature" in attrs:
                out.append(f"<key><fifths>{KEYS.get(attrs['keySignature'], 0)}</fifths></key>")
            if "timeSignature" in attrs:
                b, t = attrs["timeSignature"].split("/")
                out.append(f"<time><beats>{b}</beats><beat-type>{t}</beat-type></time>")
            if "clef" in attrs:
                sign, line = attrs["clef"][0], attrs["clef"][1:]
                out.append(f"<clef><sign>{sign}</sign><line>{line}</line></clef>")
            out.append("</attributes>")
        for kind, pitch, dur in measure:
            base, dotted = dur.rstrip("."), dur.endswith(".")
            length = int(DURATIONS[dur] * DIVISIONS)
            out.append("<note>")
            if kind == "rest":
                out.append("<rest/>")
            else:
                letter, acc, octave = parse_pitch(pitch)
                alter = f"<alter>{ALTERS[acc]}</alter>" if ALTERS[acc] else ""
                out.append(f"<pitch><step>{letter}</step>{alter}<octave>{octave}</octave></pitch>")
            out.append(f"<duration>{length}</duration><type>{XML_TYPE[base]}</type>")
            if dotted:
                out.append("<dot/>")
            out.append("</note>")
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def _varlen(n: int) -> bytes:
    buf = [n & 0x7F]
    while n > 0x7F:
        n >>= 7
        buf.insert(0, (n & 0x7F) | 0x80)
    return bytes(buf)


def to_midi(tokens: list[str], bpm: int = 90, ticks: int = 480) -> bytes:
    track = bytearray()
    tempo = 60_000_000 // bpm
    track += b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
    wait = 0
    for _, measure in _events(tokens):
        for kind, pitch, dur in measure:
            length = int(DURATIONS[dur] * ticks)
            if kind == "rest":
                wait += length
                continue
            m = midi_number(pitch)
            track += _varlen(wait) + bytes([0x90, m, 80])
            track += _varlen(length) + bytes([0x80, m, 0])
            wait = 0
    track += _varlen(wait) + b"\xff\x2f\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
