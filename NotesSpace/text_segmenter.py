import re
from gensim.models import LdaMulticore
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import concurrent.futures
import gensim
from gensim import corpora
import plotly.express as px
import pandas as pd
import openai
import os
from dotenv import load_dotenv


def get_environment_variable(variable_name):
    load_dotenv()
    return os.getenv(variable_name)

openai.api_key = get_environment_variable("OPENAI_API_KEY")

class TextSegmenter:
    def __init__(self, split_method="spacey"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.split_method = split_method
        if split_method == "spacey":
            import spacy
            self.split_model = spacy.load("en_core_web_sm")
        elif split_method == "nnsplit":
            from nnsplit import NNSplit
            self.split_model = NNSplit.load("en")

    def text_to_json(self, json_string):
        json_string = json_string.replace("\n", '')
        match = re.search("({.*})", json_string)
        return eval(match.group(1))

    def get_notes(self, segment):

        prompt = 'without losing information, summarize the following with a heading and simple bulleted notes:\n'+segment+'\n Give your summary in the following JSON format:{"heading": heading, "summary": [bullet contents, bullet contents]}'

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=256,
                n=1,
            )
            return response["choices"][0]["text"].strip()
        except:
            return None

    def get_segments(self, corpus):
        global segments
        if self.split_method == "spacey":
            doc = self.split_model(corpus)
            segments = [sent.text for sent in doc.sents]
        elif self.split_method == "nnsplit":
            splits = self.split_model.split([corpus], )[0]
            segments = [str(sentence) for sentence in splits]
        filtered_segments = []
        for segment in segments:
            if len(segment.split()) > 20:
                filtered_segments.append(segment)
        return filtered_segments

    def get_embedding(self, text):
        embedding = self.embedding_model.encode(text)
        return embedding

    def get_embeddings(self, texts):
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_embedding, text) for text in texts
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    embedding = future.result()
                except Exception as e:
                    print(f"Error: {e}")
                else:
                    embeddings.append(embedding)
        return embeddings

    def measure_similarity(self, embeddings):
        sim = []
        for i in range(len(embeddings)):
            comparison_index = i + 1
            if comparison_index < len(embeddings):
                sim.append(
                    cos_sim(embeddings[i], embeddings[comparison_index]))
            else:
                sim.append(0)
        return sim

    def find_peaks_in_similarity(self, sim):
        peaks, _ = find_peaks(np.asarray(sim[:-1]) * -1,
                              prominence=0.1, width=2)
        return peaks

    def plot_similarity(self, vid, peaks):
        plt.figure().set_figheight(2)
        plt.figure().set_figwidth(20)
        plt.plot(vid.index, vid["similarity"])
        plt.ylim((0, 1))
        plt.plot(peaks, vid.loc[peaks]["similarity"], "x")
        plt.show()

    def segment_text(self, corpus, plot=False):
        segments = self.get_segments(corpus)
        embeddings = self.get_embeddings(segments)
        sim = self.measure_similarity(embeddings)
        print(len(segments))
        print(len(embeddings))
        print(len(sim))
        vid = pd.DataFrame(
            {"segment": segments, "embedding": embeddings, "similarity": sim}
        )
        peaks = self.find_peaks_in_similarity(sim)
        if plot:
            self.plot_similarity(vid, peaks)
        dfs = []
        last_check = 0
        for ind in peaks:
            dfs.append(vid.loc[last_check:ind - 1])
            last_check = ind
        segmented_by_similarity_change = [
            " ".join(df["segment"].values) for df in dfs
        ]
        return segmented_by_similarity_change

    def extract_topics(self, text, num_topics=5, num_words=5):
        text_data = text.split('.')
        # Tokenize the text data
        tokenized_text = [gensim.utils.simple_preprocess(sentence) for sentence in text_data]

        # Remove stop words
        stop_words = gensim.parsing.preprocessing.STOPWORDS
        tokenized_text = [[word for word in sentence if word not in stop_words and word not in common_words] for sentence in tokenized_text]

        # Create a dictionary of the tokenized text data
        dictionary = corpora.Dictionary(tokenized_text)

        # Create a bag-of-words representation of the text data
        bow_corpus = [dictionary.doc2bow(text) for text in tokenized_text]

        # Train an LDA model on the bag-of-words corpus
        lda_model = LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary, passes=10, workers=2)

        # Extract the major topics from the text data
        topics = lda_model.show_topics(num_topics=-1, formatted=False)
        return topics

    def plot_topics(self, topics):
        # create a list of all the words in the topics
        for topic in topics:
            df = pd.DataFrame(topic[1], columns=['word', 'freq'])
            fig = px.scatter(df, x='word', y='freq', size='freq', title=f'Topic Keywords')
            return fig


common_words = ["uh", "um", "yeah", "okay", "well", "haha", "hmm", "gonna", "the","of","to","and","a","in","is","it","you","that","he","was","for","on","are","with","as","I","his","they","be","at","one","have","this","from","or","had","by","not","word","but","what","some","we","can","out","other","were","all","there","when","up","use","your","how","said","an","each","she","which","do","their","time","if","will","way","about","many","then","them","write","would","like","so","these","her","long","make","thing","see","him","two","has","look","more","day","could","go","come","did","number","sound","no","most","people","my","over","know","water","than","call","first","who","may","down","side","been","now","find","any","new","work","part","take","get","place","made","live","where","after","back","little","only","round","man","year","came","show","every","good","me","give","our","under","name","very","through","just","form","sentence","great","think","say","help","low","line","differ","turn","cause","much","mean","before","move","right","boy","old","too","same","tell","does","set","three","want","air","well","also","play","small","end","put","home","read","hand","port","large","spell","add","even","land","here","must","big","high","such","follow","act","why","ask","men","change","went","light","kind","off","need","house","picture","try","us","again","animal","point","mother","world","near","build","self","earth","father","head","stand","own","page","should","country","found","answer","school","grow","study","still","learn","plant","cover","food","sun","four","between","state","keep","eye","never","last","let","thought","city","tree","cross","farm","hard","start","might","story","saw","far","sea","draw","left","late","run","don't","while","press","close","night","real","life","few","north","open","seem","together","next","white","children","begin","got","walk","example","ease","paper","group","always","music","those","both","mark","often","letter","until","mile","river","car","feet","care","second","book","carry","took","science","eat","room","friend","began","idea","fish","mountain","stop","once","base","hear","horse","cut","sure","watch","color","face","wood","main","enough","plain","girl","usual","young","ready","above","ever","red","list","though","feel","talk","bird","soon","body","dog","family","direct","pose","leave","song","measure","door","product","black","short","numeral","class","wind","question","happen","complete","ship","area","half","rock","order","fire","south","problem","piece","told","knew","pass","since","top","whole","king","space","heard","best","hour","better","true","during","hundred","five","remember","step","early","hold","west","ground","interest","reach","fast","verb","sing","listen","six","table","travel","less","morning","ten","simple","several","vowel","toward","war","lay","against","pattern","slow","center","love","person","money","serve","appear","road","map","rain","rule","govern","pull","cold","notice","voice","unit","power","town","fine","certain","fly","fall","lead","cry","dark","machine","note","wait","plan","figure","star","box","noun","field","rest","correct","able","pound","done","beauty","drive","stood","contain","front","teach","week","final","gave","green","oh","quick","develop","ocean","warm","free","minute","strong","special","mind","behind","clear","tail","produce","fact","street","inch","multiply","nothing","course","stay","wheel","full","force","blue","object","decide","surface","deep","moon","island","foot","system","busy","test","record","boat","common","gold","possible","plane","stead","dry","wonder","laugh","thousand","ago","ran","check","game","shape","equate","hot","miss","brought","heat","snow","tire","bring","yes","distant","fill","east","paint","language","among","grand","ball","yet","wave","drop","heart","am","present","heavy","dance","engine","position","arm","wide","sail","material","size","vary","settle","speak","weight","general","ice","matter","circle","pair","include","divide","syllable","felt","perhaps","pick","sudden","count","square","reason","length","represent","art","subject","region","energy","hunt","probable","bed","brother","egg","ride","cell","believe","fraction","forest","sit","race","window","store","summer","train","sleep","prove","lone","leg","exercise","wall","catch","mount","wish","sky","board","joy","winter","sat","written","wild","instrument","kept","glass","grass","cow","job","edge","sign","visit","past","soft","fun","bright","gas","weather","month","million","bear","finish","happy","hope","flower","clothe","strange","gone","jump","baby","eight","village","meet","root","buy","raise","solve","metal","whether","push","seven","paragraph","third","shall","held","hair","describe","cook","floor","either","result","burn","hill","safe","cat","century","consider","type","law","bit","coast","copy","phrase","silent","tall","sand","soil","roll","temperature","finger","industry","value","fight","lie","beat","excite","natural","view","sense","ear","else","quite","broke","case","middle","kill","son","lake","moment","scale","loud","spring","observe","child","straight","consonant","nation","dictionary","milk","speed","method","organ","pay","age","section","dress","cloud","surprise","quiet","stone","tiny","climb","cool","design","poor","lot","experiment","bottom","key","iron","single","stick","flat","twenty","skin","smile","crease","hole","trade","melody","trip","office","receive","row","mouth","exact","symbol","die","least","trouble","shout","except","wrote","seed","tone","join","suggest","clean","break","lady","yard","rise","bad","blow","oil","blood","touch","grew","cent","mix","team","wire","cost","lost","brown","wear","garden","equal","sent","choose","fell","fit","flow","fair","bank","collect","save","control","decimal","gentle","woman","captain","practice","separate","difficult","doctor","please","protect","noon","whose","locate","ring","character","insect","caught","period","indicate","radio","spoke","atom","human","history","effect","electric","expect","crop","modern","element","hit","student","corner","party","supply","bone","rail","imagine","provide","agree","thus","capital","won't","chair","danger","fruit","rich","thick","soldier","process","operate","guess","necessary","sharp","wing","create","neighbor","wash","bat","rather","crowd","corn","compare","poem","string","bell","depend","meat","rub","tube","famous","dollar","stream","fear","sight","thin","triangle","planet","hurry","chief","colony","clock","mine","tie","enter","major","fresh","search","send","yellow","gun","allow","print","dead","spot","desert","suit","current","lift","rose","continue","block","chart","hat","sell","success","company","subtract","event","particular","deal","swim","term","opposite","wife","shoe","shoulder","spread","arrange","camp","invent","cotton","born","determine","quart","nine","truck","noise","level","chance","gather","shop","stretch","throw","shine","property","column","molecule","select","wrong","gray","repeat","require","broad","prepare","salt","nose","plural","anger","claim","continent","oxygen","sugar","death","pretty","skill","women","season","solution","magnet","silver","thank","branch","match","suffix","especially","fig","afraid","huge","sister","steel","discuss","forward","similar","guide","experience","score","apple","bought","led","pitch","coat","mass","card","band","rope","slip","win","dream","evening","condition","feed","tool","total","basic","smell","valley","nor","double","seat","arrive","master","track","parent","shore","division","sheet","substance","favor","connect","post","spend","chord","fat","glad","original","share","station","dad","bread","charge","proper","bar","offer","segment","slave","duck","instant","market","degree","populate","chick","dear","enemy","reply","drink","occur","support","speech","nature","range","steam","motion","path","liquid","log","meant","quotient","teeth","shell","neck"]