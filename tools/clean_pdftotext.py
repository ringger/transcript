#!/usr/bin/env python3
"""Clean pdftotext output of an NYT transcript PDF (printed from browser).

Strips repeated page headers/footers, promo blocks, newsletter signups,
production credits, and NYT boilerplate. Joins hard line breaks within
paragraphs so each paragraph is a single line.

Limitation: pdftotext cannot recover bold formatting (CSS font-weight).
In NYT interview transcripts, speaker labels like "Ezra Klein:" are
rendered bold via CSS but appear as plain text in the PDF. This script
cannot restore speaker labels that were lost during PDF generation.

Usage:
    pdftotext -layout input.pdf raw.txt
    python3 clean_pdftotext.py raw.txt cleaned.txt
"""

import re
import sys


def strip_boilerplate(text: str) -> str:
    """Remove NYT page chrome, promos, and end matter."""

    # Remove form feed characters (page breaks from pdftotext)
    text = text.replace('\x0c', '')

    # Page headers: 'date, time   Opinion | Title - The New York Times'
    text = re.sub(
        r'\d+/\d+/\d+, \d+:\d+ [AP]M\s+Opinion \|[^\n]+- The New York Times\n',
        '', text,
    )

    # Page footers: full nytimes URL + page number
    text = re.sub(
        r'https://www\.nytimes\.com/[^\s]+\s+\d+/\d+\n',
        '', text,
    )

    # Standalone nytimes URLs (may appear at start of page without page number)
    text = re.sub(r'^\s*https://www\.nytimes\.com/[^\s]+\s*\n', '', text, flags=re.MULTILINE)
    # URLs joined to text on same line (after paragraph joining or in raw)
    text = re.sub(r'https://www\.nytimes\.com/\S+\s*', '', text)

    # "More to read for free" promo line
    text = re.sub(r'More to read for free\.[^\n]*\n', '', text)

    # Promo article headlines block: multi-column layout with "MIN READ" and
    # "OPINION" markers. These appear as a block of lines with heavy leading
    # indentation (20+ spaces) that contain headline text in column layout.
    # Match lines with 20+ leading spaces that aren't transcript blockquotes.
    text = re.sub(r'^[ ]{20,}[^\n]*$', '', text, flags=re.MULTILINE)
    # Also remove standalone OPINION labels and MIN READ markers
    text = re.sub(r'^[^\n]*\d+ MIN READ[^\n]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*OPINION\s*$', '', text, flags=re.MULTILINE)

    # Newsletter signup blocks
    text = re.sub(
        r'\s*Sign up for the [^\n]+ newsletter[^\n]*\n(?:[^\n]*\n)*?[^\n]*Get it sent to your inbox\.\n',
        '\n', text,
    )

    # Title block at top (show name, headline, date, byline)
    text = re.sub(r'^\s*THE EZRA KLEIN SHOW\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*By Ezra Klein\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Produced by \w+ \w+\s*\n', '', text, flags=re.MULTILINE)
    # Date line like "Feb. 24, 2026"
    text = re.sub(r'^\s*[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}\s*\n', '', text, flags=re.MULTILINE)
    # Standalone article title (already in the transcript text)
    text = re.sub(
        r'^\s*How Fast Will A\.I\. Agents Rip Through the Economy\?\s*\n',
        '', text, flags=re.MULTILINE,
    )

    # Production credits at end ("This episode of 'The Ezra Klein Show' was produced by...")
    text = re.sub(
        r'This episode of .The Ezra Klein Show. was produced by.*',
        '', text, flags=re.DOTALL,
    )

    # Italicized outro ("You can listen to this conversation by following...")
    text = re.sub(
        r'You can listen to this conversation by following.*?from our guests here\.\s*\n',
        '', text, flags=re.DOTALL,
    )

    # NYT end-of-page boilerplate
    text = re.sub(r'The Times is committed to publishing.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Follow the New York Times Opinion.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Ezra Klein joined Opinion in 2021.*', '', text, flags=re.DOTALL)

    return text


def normalize_quotes(text: str) -> str:
    """Replace curly (smart) quotes and apostrophes with straight equivalents.

    pdftotext preserves the Unicode curly quotes from the PDF, but for
    transcript merging we want plain ASCII quotes for consistency.
    """
    replacements = {
        '\u2018': "'",   # left single curly quote
        '\u2019': "'",   # right single curly quote / apostrophe
        '\u201C': '"',   # left double curly quote
        '\u201D': '"',   # right double curly quote
        '\u2014': '—',   # em dash (keep as-is, already fine)
    }
    for fancy, plain in replacements.items():
        text = text.replace(fancy, plain)
    return text


def join_paragraphs(text: str) -> str:
    """Join hard line breaks within paragraphs into single lines.

    pdftotext -layout breaks lines at the PDF column width (~100-120 chars).
    Paragraphs are separated by blank lines. Lines within a paragraph should
    be joined with a space. Hyphenated words split across lines are repaired.
    """
    paragraphs = []
    current = []

    for line in text.split('\n'):
        stripped = line.strip()
        if stripped == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append('')
        else:
            current.append(stripped)

    if current:
        paragraphs.append(' '.join(current))

    # Repair hyphenated words that were split across lines.
    # After joining, these appear as "word- continuation" (hyphen + space).
    # Only rejoin when the continuation is lowercase (avoids real dashes
    # before capitalized words like "A.I.- related" edge cases).
    result = '\n'.join(paragraphs)
    result = re.sub(r'(\w)- (\w)', r'\1-\2', result)

    return result


def clean(text: str) -> str:
    """Full cleaning pipeline."""
    text = strip_boilerplate(text)
    text = normalize_quotes(text)

    # Collapse runs of blank lines before paragraph joining
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = join_paragraphs(text)

    # Final cleanup: collapse any remaining blank line runs
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip() + '\n'


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <raw_input.txt> <cleaned_output.txt>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        raw = f.read()

    cleaned = clean(raw)

    with open(sys.argv[2], 'w') as f:
        f.write(cleaned)

    words = len(cleaned.split())
    print(f"Cleaned: {words} words -> {sys.argv[2]}")


if __name__ == '__main__':
    main()
