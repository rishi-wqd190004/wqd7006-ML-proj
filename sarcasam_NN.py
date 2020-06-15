import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

#empty 
friends_chars={} 
Rachel=[]
Ross=[]
Joey=[]
Chandler=[]
Phoebe=[]
Monica=[]

#read the file
with open("/home/richi/sarcasm_proj/transcripts_friends/season_all/merged.csv", "r+") as fp:
    for cnt, line in enumerate(fp):
        #print("Line {}: {}".format(cnt, line))
        if line.startswith('Rachel:'):
            Rachel.append(line[8:])
        elif line.startswith('Ross:'):
            Ross.append(line[6:])
        elif line.startswith('Monica:'):
            Monica.append(line[8:])
        elif line.startswith('Chandler:'):
            Chandler.append(line[10:])
        if line.startswith('Phoebe:'):
            Phoebe.append(line[8:])
        if line.startswith('Joey:'):
            Joey.append(line[6:])

#differentiate statements
friends_chars['RACHEL']=Rachel
friends_chars['ROSS']=Ross
friends_chars['MONICA']=Monica
friends_chars['PHOEBE']=Phoebe
friends_chars['CHANDLER']=Chandler
friends_chars['JOEY']=Joey

#converting to dataframe
df1 = pd.DataFrame(friends_chars['CHANDLER'])
df2 = pd.DataFrame(friends_chars['JOEY'])
df3 = pd.DataFrame(friends_chars['PHOEBE'])
df4 = pd.DataFrame(friends_chars['RACHEL'])
df5 = pd.DataFrame(friends_chars['ROSS'])
df6 = pd.DataFrame(friends_chars['MONICA'])

#list of characters
listOfCharacters1 = [True] *df1.shape[0]
listOfCharacters2 = [False] *df2.shape[0]
listOfCharacters3 = [False] *df3.shape[0]
listOfCharacters4 = [False] *df4.shape[0]
listOfCharacters5 = [False] *df5.shape[0]
listOfCharacters6 = [False] *df6.shape[0]

#for chandler
df1['Chandler'] = listOfCharacters1
df2['Chandler'] = listOfCharacters2 
df3['Chandler'] = listOfCharacters3
df4['Chandler'] = listOfCharacters4
df5['Chandler'] = listOfCharacters5
df6['Chandler'] = listOfCharacters6

#renaming
df1=df1.rename(columns={0: 'Dialogue'})
df2=df2.rename(columns={0: 'Dialogue'})
df3=df3.rename(columns={0: 'Dialogue'})
df4=df4.rename(columns={0: 'Dialogue'})
df5=df5.rename(columns={0: 'Dialogue'})
df6=df6.rename(columns={0: 'Dialogue'})

#combining
df = pd.concat([df1, df2,df3,df4,df5,df6])
df = df.sample(frac=1).reset_index(drop=True)
print(df.head(10))

#check where its only chandler
# while :
# 	pass

#cleaning the df
#will be done later

# cols = ['Character', 'Dialogue']
# df = df[cols]
# df = df[pd.notnull(df['Dialogue'])]
# df.columns = ['Character', 'Dialogue']
# df['category_id'] = df['Character'].factorize()[0]
# category_id_df = df[['Character', 'category_id']].drop_duplicates().sort_values('category_id')
# category_to_id = dict(category_id_df.values)
# id_to_category = dict(category_id_df[['Chandler', 'Dialogue']].values)
# print(df.head(5))

#stopwords
df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['Dialogue'] = df['Dialogue'].apply(clean_text)
df['Dialogue'] = df['Dialogue'].str.replace('\d+', '')
#print(df.head(10))

#tokenizer
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 120
EMBEDDING_DIM = 64
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Dialogue'].values)
word_index = tokenizer.word_index
#print('found %s unique tokens.' % len(word_index))
#print(word_index)

x = tokenizer.texts_to_sequences(df['Dialogue'].values)
x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
print('shape of tensor:', x.shape)

y = pd.get_dummies(df['Chandler']).values
print('shape of tensor: ', y.shape)

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

#model
model1 = Sequential()
model1.add(layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
model1.add(layers.SpatialDropout1D(0.2))
model1.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model1.add(layers.Dense(2, activation='softmax'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)], verbose=1)

accr = model1.evaluate(x_test, y_test)

y_pred = model1.predict(x_test)
print(model1.summary)
print('loss: {0}\n accuracy: {1}'.format(accr[0],accr[1]))

#model 2
# model2 = Sequential()
# model2.add(layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
# model2.add(layers.Bidirectional(layers.LSTM(64,return_sequences=True)))
# model2.add(layers.Bidirectional(layers.LSTM(32)))
# model2.add(layers.Dense(64, activation='relu'))
# model2.add(layers.Dense(2, activation='sigmoid'))

# epochs = 10
# batch_size = 64

# model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history_model2 = model2.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# accr_2 = model2.evaluate(x_test, y_test)

# y_pred_2 = model2.predict(x_test)
# print('loss: {1}\n accuracy: {2}'.format(accr_2[0],accr_2[1]))

#predicting the percentage of sarcasm
#input_string = str(input())
#model1.test_on_batch(input_string, y=sarcastic or not)
input_string = str(input()) #' What rule? There\'s no rule, if anything, you owe me a table!'str(input())
df = pd.DataFrame({'Dialogue': [input_string]})
x = tokenizer.texts_to_sequences(df['Dialogue'].values)
x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
#print('shape of tensor:', x.shape)
#print(x)
#print(type(x))
y_pred = model1.predict(x)
print('y_pred: {0} Percentage of sarcasm {1} and percentage of non-sarcasm {2}',format(y_pred, y_pred[1], y_pred[0]))
