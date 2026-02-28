#!/usr/bin/env python3
"""Extract transcript from a saved NYT interview HTML page.

Recovers speaker labels by detecting bold (<strong>) formatting:
- Fully bold paragraphs = interviewer (Ezra Klein) questions
- Paragraphs starting with bold "Name:" = that speaker
- Non-bold paragraphs = continuation of previous speaker
- After an interviewer question, next non-bold = interviewee

Also strips end-of-article boilerplate (production credits, etc.)
and normalizes curly quotes to straight ASCII equivalents.

Usage:
    python3 extract_nyt_html.py input.html output.txt
"""

import html
import re
import sys
from html.parser import HTMLParser


class NYTTranscriptExtractor(HTMLParser):
    """Parse saved NYT HTML, extracting <p> text with bold annotation."""

    def __init__(self):
        super().__init__()
        self.article_depth = 0
        self.in_strong = False
        self.in_p = False
        self.skip_depth = 0
        self.current_parts = []
        self.paragraphs = []
        self.tag_stack = []

    @property
    def in_article(self):
        return self.article_depth > 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tag_stack.append(tag)
        if tag == 'section':
            if attrs_dict.get('name') == 'articleBody':
                self.article_depth = 1
            elif self.in_article:
                self.article_depth += 1
        if tag in ('script', 'style', 'figure', 'figcaption', 'button',
                    'nav', 'aside', 'source', 'picture', 'img', 'svg'):
            self.skip_depth += 1
        if not self.in_article or self.skip_depth:
            return
        if tag == 'p':
            self.in_p = True
            self.current_parts = []
        elif tag in ('strong', 'b'):
            self.in_strong = True

    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag in ('script', 'style', 'figure', 'figcaption', 'button',
                    'nav', 'aside', 'source', 'picture', 'img', 'svg'):
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag in ('strong', 'b'):
            self.in_strong = False
        elif tag == 'p' and self.in_p and self.in_article:
            self.in_p = False
            full_text = ''.join(t for t, _ in self.current_parts).strip()
            if not full_text:
                return
            bold_chars = sum(len(t) for t, b in self.current_parts if b)
            total_chars = sum(len(t) for t, _ in self.current_parts)
            mostly_bold = total_chars > 0 and bold_chars > total_chars * 0.8
            self.paragraphs.append((full_text, mostly_bold))
            self.current_parts = []
        elif tag == 'section' and self.in_article:
            self.article_depth -= 1

    def handle_data(self, data):
        if self.in_article and self.in_p and not self.skip_depth:
            self.current_parts.append((data, self.in_strong))

    def handle_entityref(self, name):
        char = html.unescape(f"&{name};")
        self.handle_data(char)

    def handle_charref(self, name):
        char = html.unescape(f"&#{name};")
        self.handle_data(char)


def assign_speakers(paragraphs, interviewer="Ezra Klein", interviewee="Jack Clark"):
    """Assign speakers based on bold pattern and explicit labels.

    NYT convention for interview transcripts:
    - Interviewer's monologue precedes the first explicit label
    - Bold paragraphs = interviewer (questions/comments)
    - "Name:" explicit prefix in bold = that speaker
    - Non-bold after interviewer question = interviewee answering
    - Non-bold continuation = same speaker as previous
    """
    result = []
    current_speaker = interviewer
    interview_started = False

    for text, is_bold in paragraphs:
        if text.startswith(f"{interviewer}:"):
            current_speaker = interviewer
            interview_started = True
        elif text.startswith(f"{interviewee}:"):
            current_speaker = interviewee
            interview_started = True
        elif not interview_started:
            current_speaker = interviewer
        elif is_bold:
            current_speaker = interviewer
        else:
            if current_speaker == interviewer:
                current_speaker = interviewee
        result.append((text, current_speaker))

    return result


def strip_end_matter(paragraphs):
    """Remove production credits, boilerplate, and outro after transcript ends."""
    cutoff_patterns = [
        "You can listen to this conversation by following",
        "This episode of",
        "The Times is committed",
        "Follow the New York Times",
    ]
    result = []
    for text, speaker in paragraphs:
        if any(text.startswith(p) for p in cutoff_patterns):
            break
        result.append((text, speaker))
    return result


def normalize_quotes(text):
    """Replace curly (smart) quotes and apostrophes with straight equivalents."""
    return (text
            .replace('\u2018', "'").replace('\u2019', "'")
            .replace('\u201C', '"').replace('\u201D', '"'))


def format_transcript(labeled_paragraphs):
    """Format as plain text with speaker labels on turn changes."""
    lines = []
    prev_speaker = None
    for text, speaker in labeled_paragraphs:
        if speaker != prev_speaker:
            if not text.startswith(f"{speaker}:"):
                text = f"{speaker}: {text}"
        lines.append(text)
        lines.append("")
        prev_speaker = speaker
    return "\n".join(lines).strip() + "\n"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.html> <output.txt>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        html = f.read()

    parser = NYTTranscriptExtractor()
    parser.feed(html)

    labeled = assign_speakers(parser.paragraphs)
    labeled = strip_end_matter(labeled)

    output = format_transcript(labeled)
    output = normalize_quotes(output)

    with open(sys.argv[2], 'w') as f:
        f.write(output)

    words = len(output.split())
    ezra = len(re.findall(r'^Ezra Klein:', output, re.MULTILINE))
    jack = len(re.findall(r'^Jack Clark:', output, re.MULTILINE))
    print(f"Extracted: {words} words -> {sys.argv[2]}")
    print(f"Speaker labels: Ezra Klein: {ezra}, Jack Clark: {jack}")


if __name__ == '__main__':
    main()
