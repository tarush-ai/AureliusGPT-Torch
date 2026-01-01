import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#preprocessing
greekwords = {'ἔμφρων': 'emphron', 'σύμφρων': 'sumphron', 'ὑπέρφρων': 'huperphron', 'Νόμος': 'nomos', 'νέμων': 'nemon', 'ἀξιοπίστως': 'axiopistos', 'θεοφόρητος': 'theophoretos', 'οἰκονομίαν': 'oikonomian', 'τοῦτο': 'touto', 'ἔφερεν': 'eferen', 'αὐτῷ': 'auto', 'εὔμοιρος': 'eumoiros', 'εὐδαιμονία': 'eudaimonia', 'εὐπατρίδαι': 'eupatridai', 'καθότι': 'kathoti', 'κατορθώσεως': 'katorthoseos', 'κόσμος': 'kosmos', 'μέλος': 'melos', 'μέρος': 'meros', 'παρειλήφαμεν': 'pareilephamen', 'συμβαίνειν': 'symbainein', 'τάσις': 'tasis', 'ἀγαθός': 'agathos', 'ἀκτῖνες': 'aktines', 'ἐκτείνεσθαι': 'ekteinesthai', 'δαίμων': 'daimon', 'κατορθώσεις': 'katorthoseis', 'ἀγαθὸς': 'agathos', 'ἀυτῷ': 'auto'}

vocab_size = 1000
batch_size = 512