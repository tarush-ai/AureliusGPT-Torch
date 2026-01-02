import re, os, unicodedata
from config import PROJECT_ROOT, greekwords

class SyntheticFormatter:
    def __init__(self):
        pass

    def process(self, text):
        meditations = unicodedata.normalize("NFC", text)

        try:
            startindex = meditations.index("THE FIRST BOOK")
            meditations = meditations[startindex:]
            endindex = meditations.index("APPENDIX")
            meditations = meditations[:endindex] 
        except ValueError:
            pass

        book_name = r"THE\s+[A-Z]+\s+BOOK\s+[IVXLCDM]+\."
        section_name = r"\n\n[IVXLCDM]+\. "
        underline = r"[_]+"
        newline_in_sentence = r"(?<!\n)\n(?!\n)"

        for key, value in greekwords.items():
            meditations = meditations.replace(key, value)

        meditations = re.sub(newline_in_sentence, " ", meditations)
        meditations = re.sub(book_name, "\n\n", meditations)
        meditations = re.sub(section_name, "\n\n", meditations)
        meditations = re.sub(underline, "", meditations)
        meditations = re.sub(r'\n{3,}', '\n\n', meditations)
        
        meditations = meditations.strip()

        final_sections = [s.strip() for s in meditations.split('\n\n') if s.strip()]
        
        return meditations, final_sections

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

