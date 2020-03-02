
import numpy as np
import tensorflow as tf
import math
import cv2
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from pathlib import Path
import pickle
import pandas as pd
import requests
TEMPLATES_AUTO_RELOAD = True

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


###############################################################################################
######################### MAIN REDIRECTION ROUTES #############################################
############################################################################################


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/mlProjIndex')
def mlProjIndex():
    return render_template('mlProjects/index.html')

@app.route('/dlProjIndex')
def dlProjIndex():
    return render_template('dlProjects/index.html')

@app.route('/nlpProjIndex')
def nlpProjIndex():
    return render_template('nlpProjects/index.html')


###############################################################################################
######################### MACHINE LEARNING PROJECTS #############################################
############################################################################################

#--------------------------
#--- GRAD ADMISSION -------
#--------------------------



@app.route('/grad_admissions')
def grad_admissions():
    return render_template('mlProjects/graduateAdmissions.html')


@app.route('/grad_predict',methods=['POST'])
def grad_predict():

    int_features = [float(x) for x in request.form.values()]


    if int_features[6] == 1.0:
        del int_features[6]
        # int_features.insert(0, 0)
        # int_features.insert(1, 1)
        int_features.append(0)
        int_features.append(1)
    else:
        del int_features[6]
        # int_features.insert(0, 1)
        # int_features.insert(1, 0)
        int_features.append(1)
        int_features.append(0)

    dfFeat = pd.DataFrame(int_features).transpose()
    #D:\herokuDemo\savedModels\ml\grad_adm
    with open('savedModels/ml/grad_adm/ssFit.pkl', 'rb') as file:

        scaler = pickle.load(file)



    #df = pd.read_csv('Data/ml/grad_adm/Admission_Predict.csv')
    #X = df.iloc[:, 1:8]

    #from sklearn.preprocessing import OneHotEncoder
    #enc = OneHotEncoder(categorical_features=[6])
    #X = enc.fit_transform(X).toarray()

    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #X = scaler.fit(X)

    # return render_template('mlProjects/graduateAdmissions.html',
    #                        prediction_text='With Provided Data Your Chances Of Getting Admission is  {}'.format(
    #                            dfFeat))
    dfFeat = scaler.transform(dfFeat)

    dfFeat = pd.DataFrame(dfFeat)
    
    pkl_Filename = 'savedModels/ml/grad_adm/regModel'


    with open('savedModels/ml/grad_adm/regModel', 'rb') as file:

        Pickled_LR_Model = pickle.load(file)

    prediction = round(Pickled_LR_Model.predict(dfFeat)[0]*100,2)
    #prediction = Pickled_LR_Model.predict(dfFeat)

    return render_template('mlProjects/graduateAdmissions.html',prediction_text='With Provided Data Your Chances Of Getting Admission is  {}'.format(prediction))

###############################################################################################
######################### DEEP LEARNING PROJECTS #############################################
############################################################################################

#--------------------------
#--- intel classification -------
#--------------------------


@app.route('/intelClassify')
def intelClassify():
    return render_template('dlProjects/intelClassify.html')

@app.route('/intel_pred', methods = ['GET', 'POST'])
def intel_pred():
    if request.method == 'POST':
        f = request.files['file']
        ext = Path(f.filename).suffix
        f.filename = 'image'+ext
        model_intel = tf.keras.models.load_model("testing.h5")

        f.save(f.filename)
        image = cv2.imread(f.filename)
        image = cv2.resize(image, (150, 150))
        image = np.array(image)
        image = np.reshape(image, (1, 150, 150, 3))
        #class_code = model_intel.predict_proba((image, tf.float32))
        
        labels = {2: 'glacier', 4: 'sea', 0: 'buildings', 1: 'forest', 5: 'street', 3: 'mountain'}
        class_code = model_intel.predict_classes((image, tf.float32))[0]
        #class_code = model_intel.predict((image, tf.float32))[0].argsort()[-5:][::-1]
        res = labels[class_code]
        return render_template('dlProjects/intelClassify.html',prediction_text='image given is of  {}'.format(res))

#----------------------------------#
#--------- butterfly classify -----#
#----------------------------------#
@app.route('/butterflyClassify')
def butterflyClassify():
    return render_template('dlProjects/butterflyClassify.html')


@app.route('/butterfly_pred', methods=['GET', 'POST'])
def butterfly_pred():
    if request.method == 'POST':
        f = request.files['file']
        ext = Path(f.filename).suffix
        f.filename = 'image' + ext
        model = tf.keras.models.load_model("savedModels/dl/my_model.h5")
        #D:\herokuDemo\savedModels\dl
        f.save(f.filename)

        def imageWork(location):
            image = cv2.imread(location)
            image = cv2.resize(image, (120, 120))
            image = image / 255
            image = np.array(image)
            image = np.reshape(image, (1, 120, 120, 3))
            print(image.shape)
            print(image.max())
            print(type(image))
            return image

        image = imageWork(f.filename)

        species = {7: 'Giant Swallowtail',
                   2: 'Zebra Longwing',
                   6: 'Mourning Cloak',
                   3: 'Crimson-patched Longwing',
                   5: 'American Copper',
                   0: 'Painted Lady',
                   9: 'Red Admiral',
                   4: 'Common Buckeye',
                   8: 'Cabbage White',
                   1: 'Monarch'}

        res = model.predict(image)
        #return render_template('dlProjects/butterflyClassify.html', prediction_text=image.shape)
        #type(res[0])
        val = np.argmax(res[0])
        result = species[val]

        return render_template('dlProjects/butterflyClassify.html', prediction_text='image of butterfly given belongs to '+result+' species.')

#---------------------------------------------------------------------------------------#
#--------------------------- FLOWER CLASSIFICATION -------------------------------------#
#---------------------------------------------------------------------------------------#

@app.route('/flowerClassify')
def flowerClassify():
    return render_template('dlProjects/flowerClassify.html')

@app.route('/flower_pred', methods=['GET', 'POST'])
def flower_pred():
    if request.method == 'POST':
        f = request.files['file']
        ext = Path(f.filename).suffix
        f.filename = 'image' + ext
        model = tf.keras.models.load_model("savedModels/dl/flowerDetection/my_model.h5")
        #D:\herokuDemo\savedModels\dl
        f.save(f.filename)

        def imageWork(location):
            image = cv2.imread(location)
            image = cv2.resize(image, (120, 120))
            image = image / 255
            image = np.array(image)
            image = np.reshape(image, (1, 120, 120, 3))

            return image

        image = imageWork(f.filename)

        species = ['Tulip', 'Daisy', 'Sunflower', 'Rose', 'Dandelion']

        res = model.predict(image)
        #return render_template('dlProjects/butterflyClassify.html', prediction_text=image.shape)
        #type(res[0])
        val = np.argmax(res[0])
        result = species[val]

        return render_template('dlProjects/flowerClassify.html', prediction_text='Image of flower given belongs to '+result+' Category.')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


###########################################################################################################
######################################### NLP PROJECTS ####################################################
###########################################################################################################
#------------------- BBC ARTICLE CLASSIFICATION --------------------------#
@app.route('/articleClassify')
def articleClassify():
    return render_template('nlpProjects/articleClassify.html')

@app.route('/article_predict',methods=['POST'])
def article_predict():

    import re

    text = [x for x in request.form.values()]
    text = text[0]
#D:\herokuDemo\savedModels\nlp\articleClassification
    with open('savedModels/nlp/articleClassification/leEnc.pkl', 'rb') as file:
        leEnc = pickle.load(file)

    with open('savedModels/nlp/articleClassification/tvEnc.pkl', 'rb') as file:
        tvEnc = pickle.load(file)

    with open('savedModels/nlp/articleClassification/stop_words.pkl', 'rb') as file:
        stop_words = pickle.load(file)

    with open('savedModels/nlp/articleClassification/stemmer.pkl', 'rb') as file:
        stemmer = pickle.load(file)

    with open('savedModels/nlp/articleClassification/lemmatizer.pkl', 'rb') as file:
        lemmatizer = pickle.load(file)

    with open('savedModels/nlp/articleClassification/classModel', 'rb') as file:
        model = pickle.load(file)

    def preprocess(text):
        text = text.lower()
        text = re.sub('[^a-zA-Z ]', '', text)
        # removing stop words
        wordsList = text.split()
        newWordsList = []
        for word in wordsList:
            if word not in stop_words:  # remove stop words
                word = stemmer.stem(word)  # using porter stemmer
                word = lemmatizer.lemmatize(word)
                newWordsList.append(word)

        return " ".join(newWordsList)

    preData = preprocess(text)
    finalSample = tvEnc.transform([preData])
    result = model.predict(finalSample)
    #3 = > sport, 4 = > tech, 0 = > business, 1 = > entertainment, 2 = > politics
    catList = ['Business','Entertainment','Politics','Sport','Technology']

    return render_template('nlpProjects/articleClassify.html', prediction_text=' Article belongs to '+catList[result[0]]+' category.')

#------------------- BBC ARTICLE CLASSIFICATION --------------------------#
@app.route('/disasterTweet')
def disasterTweet():
    return render_template('nlpProjects/disasterTweet.html')

@app.route('/disasterTweet_predict',methods=['POST'])
def disasterTweet_predict():
    import re
    from bs4 import BeautifulSoup
    from nltk import word_tokenize
    import string
    from nltk.corpus import stopwords
    from nltk import WordNetLemmatizer

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_url(text):
        # remove urls
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_html(w):
        soup = BeautifulSoup(w)
        text = soup.get_text()
        return w

    def cleanData(data):
        # remove urls
        data['text'] = data['text'].apply(lambda x: remove_url(x))

        # remove emojis
        data['text'] = data['text'].apply(lambda x: remove_emoji(x))

        # tokenizing words
        data['text'] = data['text'].apply(lambda x: word_tokenize(x))

        # convert all text to lowercase
        data['text'] = data['text'].apply(lambda x: [w.lower() for w in x])

        # remove html tags
        data['text'] = data['text'].apply(lambda x: [remove_html(w) for w in x])

        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))

        # removing puncutations
        data['text'] = data['text'].apply(lambda x: [re_punc.sub('', w) for w in x])

        # removing non alphabetic words
        data['text'] = data['text'].apply(lambda x: [w for w in x if w.isalpha()])

        # removing stopwords
        data['text'] = data['text'].apply(lambda x: [w for w in x if w not in stopwords.words('english')])

        # removing short words
        data['text'] = data['text'].apply(lambda x: [w for w in x if len(w) > 2])
        return data

    text = [x for x in request.form.values()]
    text = text[0]

    dataTrial = pd.DataFrame([text], columns=['text'])
    dataTrial = cleanData(dataTrial)

    lem = WordNetLemmatizer()
    dataTrial['text'] = dataTrial['text'].apply(lambda x: [lem.lemmatize(w) for w in x])
    dataTrial['text'] = dataTrial['text'].apply(lambda x: ' '.join(x))

    with open('savedModels/nlp/disasterTweetClassification/regModel', 'rb') as file:
        model = pickle.load(file)

    with open('savedModels/nlp/disasterTweetClassification/tfidfFit', 'rb') as file:
        tfidf = pickle.load(file)

    Xtest = dataTrial['text']
    Xtest = tfidf.transform(Xtest)
    Xtest = Xtest.toarray()
    y_pred = model.predict(Xtest)[0]

    if y_pred == 1:
        msg = 'Tweet is disaster related'
    else:
        msg = 'Tweet is not related to disaster'



    return render_template('nlpProjects/disasterTweet.html',prediction_text=msg)




#######################################################################################################################
############################# CONPUTER VISION #############################
###########################################################################

# ------------------- object movement tracking ------------- #

@app.route('/objectTracking')
def objectTracking():
    return render_template('cv/objectTracking.html')

@app.route('/detectMove')
def detectMove():
    corner_track_params = dict(maxCorners=10,
                               qualityLevel=0.3,
                               minDistance=7,
                               blockSize=7)

    lk_params = dict(winSize=(200, 200), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(0)

    ret, prev_frame = cap.read()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                      **corner_track_params)

    mask = np.zeros_like(prev_frame)
    num = 0
    move=False
    while True:
        ret, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                        frame_gray,
                                                        prevPts,
                                                        None,
                                                        **lk_params)
        '''
        good_new = nextPts[status == 1]
        good_prev = prevPts[status == 1]
        #move = False
        for i, (new, prev) in enumerate(zip(good_new, good_prev)):
            x_new, y_new = new.ravel()
            x_prev, y_prev = prev.ravel()

            mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev),
                            (0, 255, 0), 3)

            frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)

            # text for escaping cam
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text='press escape key to close cam', org=(50, 200), fontFace=font, fontScale=1,
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # detecting movement
            dist = math.sqrt((x_new - x_prev) ** 2 + (y_new - y_prev) ** 2)
            if dist > 5:
                # cv2.imwrite('opencv'+str(num)+'.jpg',frame)
                num = num + 1
                move = True
                # print('movement detected')
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(frame, text='Movement Detected', org=(50, 400), fontFace=font, fontScale=2,
                #            color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        if move==True:
            cv2.putText(frame, text='Movement Detected', org=(50, 400), fontFace=font, fontScale=2,
                    color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        '''
        img = cv2.add(frame, mask)
        cv2.imshow('tracking', img)

        # dist = math.sqrt((x_new - x_prev)**2 + (y_new - y_prev)**2)

        '''if dist > 5:
            cv2.imwrite('opencv'+str(num)+'.jpg',frame)
            num = num+1'''

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

        #prev_gray = frame_gray.copy()
        #prevPts = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()
    return render_template('cv/objectTracking.html')


if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=8080)
    app.run(debug=True)