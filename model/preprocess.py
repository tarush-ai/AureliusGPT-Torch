import re, os, unicodedata
from config import PROJECT_ROOT, greekwords

class Preprocessor:
    def __init__(self):
        pass

    def process(self, text):
        meditations = unicodedata.normalize("NFC", text)

        startindex = meditations.index("THE FIRST BOOK")
        meditations = meditations[startindex:]
        endindex = meditations.index("APPENDIX")
        meditations = meditations[:endindex] 

        book_name = r"THE\s+[A-Z]+\s+BOOK\s+[IVXLCDM]+\.\s"
        section_name = r"\n\n[IVXLCDM]+\. "
        underline = r"[_]+"
        book_end = r"\n\n\n\n"
        newline_in_sentence = r"(?<!\n)\n(?!\n)"

        meditations = re.sub(book_name, "<BEGIN> \n", meditations)
        meditations = re.sub(book_end, "<END> ", meditations)
        meditations = re.sub(section_name, "\n <END> \n <BEGIN> \n", meditations)
        meditations = re.sub(underline, "", meditations)
        meditations = re.sub(newline_in_sentence, " ", meditations)

        for key, value in greekwords.items():
            meditations = meditations.replace(key, value)
        
        
        split_pattern = f"{book_name}|{section_name}"
        raw_sections = re.split(split_pattern, meditations)

        final_sections = []
        for section in raw_sections:
            if not section.strip():
                continue
            sentences = self.split_into_sentences(section)
            if sentences:
                final_sections.append(sentences)
        
        return meditations, final_sections

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

process = Preprocessor()
with open(os.path.join(PROJECT_ROOT, "meditations.txt"), "r") as f:
    meditations = f.read()
p = process.process(meditations)
print(p[1])
