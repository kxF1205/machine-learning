import nltk
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import unidecode
from pycontractions import Contractions
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
SEND_DETECTOR = nltk.data.load("tokenizers/punkt/english.pickle")
import glob
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#read all the txt files' path as a list
train_txt_n = glob.glob('the path to the directory neg trainging txt')
train_txt_p = glob.glob('the path to the directory pos training txt')
test_txt_n =  glob.glob('the path to the directory of neg testing txt')
test_txt_p =  glob.glob('the path to the directory of pos testing txt')


#'neg' is 0, and 'pos' is 1
labels = ['neg','pos']

#the list of stop words
stop_words = set(stopwords.words('english'))

#list of contractions
contractions_dict = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't":"were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

#convert accented characters
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text
#lemmatize, the input is the word_tokenize
def lemmatize_sentence(wt):
    wl = WordNetLemmatizer()
    ls = []
    for word, tag in pos_tag(wt):
        if tag.startswith('NN'):
            ls.append(wl.lemmatize(word,pos = 'n'))
        elif tag.startswith('VB'):
            ls.append(wl.lemmatize(word,pos='v'))
        elif tag.startswith('JJ'):
            ls.append(wl.lemmatize(word,pos='a'))
        elif tag.startswith('R'):
            ls.append(wl.lemmatize(word,pos='r'))
        else:
            ls.append(word)
    return ls



#choose the Porter Stemming algorithm to stem word_token list
def stem_sentence(tokens):
    porter = PorterStemmer()
    stemmed = [porter.stem(w) for w in tokens]
    return stemmed

#this function is for just cleaning the data to test the model
def cleaning(file_list,c):
    # this list takes a txt as a sublist
    txt_ls = []
    # this is a label list for the txt_ls
    test_l_y = []
    for t in file_list:
        f = open(t)
        raw = f.read()
        f.close()
        # clean the HTML
        soup = BeautifulSoup(raw, 'html.parser')
        new_raw = soup.get_text(separator=" ")
        # a list of sentences in the txt file
        sents = nltk.sent_tokenize(new_raw)
        # now start cleaning in each sentence
        txt = []
        for strings in sents:
            #expand the contraction
            strings = expand_contractions(strings)
            strings = remove_accented_chars(strings)
            # token the words in the sentences
            tokens = nltk.word_tokenize(strings)
            # get the lowercased string
            words = [w.lower() for w in tokens]
            # filter out punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in words]
            if stripped != []:
                txt = txt + stripped
        t = ' '.join(txt)
        txt_ls.append(t)
        test_l_y.append(c)
    return txt_ls, test_l_y

#c is the label of the txt files, this function is for the training data
def text_preprocessing(file_list,c, remove_num=False):
    #this list takes a sentence as a sublist
    tokens_list = []
    #this is a label list for the tokens_list
    train_y = []
    #this list takes a txt as a sublist
    txt_ls= []
    #this is a label list for the txt_ls
    train_l_y=[]
    for t in file_list:
        f = open(t)
        raw = f.read()
        f.close()
        #clean the HTML
        soup = BeautifulSoup(raw,'html.parser')
        new_raw = soup.get_text(separator=" ")
        # a list of sentences in the txt file
        sents = nltk.sent_tokenize(new_raw)
        # now start cleaning in each sentence
        txt=[]
        for strings in sents:
            # expand the contraction
            strings = expand_contractions(strings)
            strings = remove_accented_chars(strings)
            # token the words in the sentences
            tokens = nltk.word_tokenize(strings)
            # use the function lemmatize_sentence to lemmatize
            tokens = lemmatize_sentence(tokens)
            # use the function stem_sentence to stem the token
            #tokens= stem_sentence(tokens)
            #get the lowercased string
            words = [w.lower() for w in tokens]
            #filter out punctuation
            table = str.maketrans('','',string.punctuation)
            stripped = [w.translate(table) for w in words]
            #you can choose to remove or not remove the numbers
            if remove_num  == False :
                stripped = [w for w in stripped if w.isalpha()]
            else:
                stripped = [w for w in stripped if w.isalpha() or w.isnumeric()]
            #remove stop words form the list
            filtered_words = [w for w in stripped if not w in stop_words]
            if filtered_words != []:
                sent = ' '.join(filtered_words)
                tokens_list.append(sent)
                train_y.append(c)
                txt=txt+filtered_words
        t = ' '.join(txt)
        txt_ls.append(t)
        train_l_y.append(c)
    return tokens_list,train_y,txt_ls,train_l_y

#it will be more convenient if we write the preprocessed data to a new csv file
def write_files(train_txt_n,train_txt_p,test_txt_n,test_txt_p):
    #this is for the training data
    #the negative list and its label
    train_n,train_y_n,train_ntt_ls,train_tt_ny =text_preprocessing(train_txt_n,0)
    #the positive list and its label
    train_p,train_y_p,train_ptt_ls,train_tt_py =text_preprocessing(train_txt_p,1)


    #the total train list, the element of the list is a sentence
    train_x = train_n+train_p
    dfx_csv = pd.DataFrame(train_x)
    dfx_csv.to_csv('define a path here')
    #the total label list
    train_y = train_y_n+train_y_p
    dfy_csv = pd.DataFrame(train_y)
    dfy_csv.to_csv('define a path here')

    #the total test list , the element of the list is a txt
    train_tt_x = train_ntt_ls+train_ptt_ls
    dftx_csv = pd.DataFrame(train_tt_x)
    dftx_csv.to_csv('define a path here')
    #the total label list
    train_tt_y = train_tt_ny+train_tt_py
    dfty_csv = pd.DataFrame(train_tt_y)
    dfty_csv.to_csv('define a path here')

    #this is for the testing data
    test_n,test_y_n = cleaning(test_txt_n,0)
    test_p,test_y_p = cleaning(test_txt_p,1)

    # the total test list, the element of the list is a txt
    test_x = test_n + test_p
    dftest_x = pd.DataFrame(test_x)
    dftest_x.to_csv('define a path here')
    # the total label list
    test_y = test_y_n + test_y_p
    dftest_y = pd.DataFrame(test_y)
    dftest_y.to_csv('define a path here')




#count, data input should be a list of strings
def count_Vec(data,maxdf=1,mindf=1,maxfeature=None):
    vectorizer = CountVectorizer(max_df=maxdf,min_df=mindf,max_features=maxfeature,analyzer='word')
    X_training = vectorizer.fit_transform(data)
    return   X_training,vectorizer
#tf
def tf_Vec(data):
    X_train_counts = count_Vec(data)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_tf
#tf_idf
def tfidf_Vec(data):
    X_train_counts = count_Vec(data)
    tf_transformer = TfidfTransformer.fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_tf


def get_frequency(file_name,outfile_path):
    n = ['index','content']
    data = pd.read_csv(file_name,names=n,header=0)
    #sparse_matrix,vectorizer = count_Vec(data)
    #print('step1')
    vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word',lowercase=False)
    #print('step2')
    sparse_matrix = vectorizer.fit_transform(data['content'])
    #print(vectorizer.get_feature_names())
    frequencies = sum(sparse_matrix).toarray()[0]
    #print('step4')
    df = pd.DataFrame(frequencies, index=vectorizer.get_feature_names(), columns=['frequency'])
    df = df.sort_values(by=['frequency'],ascending = False)
    df.to_csv('outfile_path')
    #print(df)

def draw_gram(file_name,csv_list):
    ls = csv_list
    for l in ls:
        filep = file_name+l
        data = pd.read_csv(filep,header=0)
        dataf = data[0:50]
        newName = file_name+'f50_'+l
        dataf.to_csv(newName)
        datal = data[-50:]
        newNamel = file_name+'l50_'+l
        datal.to_csv(newNamel)


