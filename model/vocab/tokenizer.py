import os, io, sys
import sentencepiece as spm
from config import PROJECT_ROOT, vocab_size
from model.vocab.preprocess import Preprocessor

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
    
    def train(self, all_sentences, model_prefix=None):
        if model_prefix is None:
            model_prefix = os.path.join(PROJECT_ROOT, "data", "tokenizer")
            
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_sentences), 
            model_prefix=model_prefix, 
            model_type="bpe",
            vocab_size=vocab_size,
            user_defined_symbols=["<BEGIN>", "<END>", "<PAD>"]
        )
        self.sp.Load(model_prefix + ".model")
    
    def load_weights(self, path):
        self.sp.Load(path)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids):
        return self.sp.DecodeIds(ids)


    def test(self, file):
        text_content = None
        if file:
            text_content = file
        
        if not text_content:
            test_file_path = os.path.join(os.path.dirname(__file__), "tokenize_test.txt")
            if os.path.exists(test_file_path):
                with open(test_file_path, "r") as f:
                    text_content = f.read()
            else:
                print(f"Default test file not found at {test_file_path}")
                return

        if text_content:
            try:
                encoded_ids = self.encode(text_content)
                decoded_text = self.decode(encoded_ids)
                
                output_content = f"Original Text:\n{text_content}\n\nToken IDs:\n{encoded_ids}\n\nDecoded Text:\n{decoded_text}\n"
                
                output_file_path = os.path.join(os.path.dirname(__file__), "tokenize_test_output.txt")
                with open(output_file_path, "w") as f:
                    f.write(output_content)
                    
                print(f"Saved to {output_file_path}.")
            except Exception as e:
                print(f"Error during tokenization test: {e}")

if __name__ == "__main__":
    file = None
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test != "test":
            print("Only permitted argument is 'test'; Please try again.")
            pass

    else:
        print("Tokenization logic is wrapped into overall training functionality.")
        pass

    if len(sys.argv) > 2:
        filepath = sys.argv[2]
        try:
            with open(filepath, "r") as f:
                file = f.read()
        except Exception as e:
            print("Invalid filepath, falling back to original test.")
            file = None

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        Tokenizer().test(file)
