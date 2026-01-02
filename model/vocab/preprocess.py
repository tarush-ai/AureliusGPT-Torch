import re, os, unicodedata
import sys
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

    def test(self, file):
        processed = None
        if file:
            try:
                processed = self.process(file)
            except Exception:
                print("The processed file is not compliant with preprocess' requirements. Falling back to default file.\n")
                processed = None
        if not processed:
            test_file_path = os.path.join(os.path.dirname(__file__), "preprocess_test.txt")
            with open(test_file_path, "r") as f:
                processed = self.process(f.read())

        output_file_path = os.path.join(os.path.dirname(__file__), "preprocess_test_output.txt")
        with open(output_file_path, "w") as f:
            f.write(processed[0])
            
        print(f"Saved to {output_file_path}.")

if __name__ == "__main__":
    file = None
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test != "test":
            print("Only permitted argument is 'test'; Please try again.")
            pass

    else:
        print("Preprocessing logic is wrapped into overall training functionality.")
        pass
    
    if len(sys.argv) > 2:
        filepath = sys.argv[2]
        try:
            with open(filepath, "r") as f:
                file = f.read()
        except Exception as e:
            print("Invalid filepath, falling back to original test.")
            file = None

    Preprocessor().test(file)




