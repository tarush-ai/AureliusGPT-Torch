import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#preprocessing
greekwords = {'ἔμφρων': 'emphron', 'σύμφρων': 'sumphron', 'ὑπέρφρων': 'huperphron', 'Νόμος': 'nomos', 'νέμων': 'nemon', 'ἀξιοπίστως': 'axiopistos', 'θεοφόρητος': 'theophoretos', 'οἰκονομίαν': 'oikonomian', 'τοῦτο': 'touto', 'ἔφερεν': 'eferen', 'αὐτῷ': 'auto', 'εὔμοιρος': 'eumoiros', 'εὐδαιμονία': 'eudaimonia', 'εὐπατρίδαι': 'eupatridai', 'καθότι': 'kathoti', 'κατορθώσεως': 'katorthoseos', 'κόσμος': 'kosmos', 'μέλος': 'melos', 'μέρος': 'meros', 'παρειλήφαμεν': 'pareilephamen', 'συμβαίνειν': 'symbainein', 'τάσις': 'tasis', 'ἀγαθός': 'agathos', 'ἀκτῖνες': 'aktines', 'ἐκτείνεσθαι': 'ekteinesthai', 'δαίμων': 'daimon', 'κατορθώσεις': 'katorthoseis', 'ἀγαθὸς': 'agathos', 'ἀυτῷ': 'auto'}

#tokenization
vocab_size = 2000

#model
num_blocks = 3
d_model = 32
h = 4
d_head = d_model // h
d_ff = 4 * d_model

#training time
batch_size = 512
num_epochs = 20
max_seq_length = batch_size
lr = 3e-4

#runtime
max_tokens = 100
temperature = 0.8