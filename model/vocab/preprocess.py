import re, os, unicodedata
from config import PROJECT_ROOT, greekwords

class Preprocessor:
    def __init__(self):
        pass

    def process(self, text):
        meditations = unicodedata.normalize("NFC", text)

        # Check if the text has the expected structure of the original Meditations
        is_original = "THE FIRST BOOK" in meditations and "APPENDIX" in meditations

        if is_original:
            startindex = meditations.index("THE FIRST BOOK")
            meditations = meditations[startindex:]
            endindex = meditations.index("APPENDIX")
            meditations = meditations[:endindex] 

            book_name = r"THE\s+[A-Z]+\s+BOOK\s+[IVXLCDM]+\.\s"
            section_name = r"\n\n[IVXLCDM]+\. "
            book_end = r"\n\n\n\n"
            
            meditations = re.sub(book_name, "<BEGIN> \n", meditations)
            meditations = re.sub(book_end, "<END> ", meditations)
            meditations = re.sub(section_name, "\n <END> \n <BEGIN> \n", meditations)
            
            split_pattern = f"{book_name}|{section_name}"
        else:
            # Minimal processing for synthetic data
            split_pattern = r"\n\n"  # Split by paragraphs or double newlines

        underline = r"[_]+"
        newline_in_sentence = r"(?<!\n)\n(?!\n)"

        meditations = re.sub(underline, "", meditations)
        meditations = re.sub(newline_in_sentence, " ", meditations)

        for key, value in greekwords.items():
            meditations = meditations.replace(key, value)
        
        raw_sections = re.split(split_pattern, meditations)

        final_sections = []
        for section in raw_sections:
            if not section.strip():
                continue
            sentences = self.split_into_sentences(section)
            if sentences:
                final_sections.extend(sentences)
        
        return meditations, final_sections

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]