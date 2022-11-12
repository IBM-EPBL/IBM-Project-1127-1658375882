from __future__ import division, print_function

import os
if __name__=='__main__':
    os.environ.setdefault('FLASK_ENV','development')
    
import numpy as np
import cv2



from tensorflow.keras.models import load_model


from flask import Flask,render_template,request
from werkzeug.utils import secure_filename


app = Flask(__name__)


MODEL_PATH = 'models/my_model.h5'



model = load_model(MODEL_PATH)

    
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model): 
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(100,100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    
    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',f.filename )  
        secure_filename(f.filename)
        f.save(file_path)

       
        preds = model_predict(file_path, model)

        
        pred_class = preds.argmax()              
 
        
        CATEGORIES = ['Leaf:Pepper bell-->Disease:Bacterial spot-->Suggestions:Washing seeds for 40 minutes in diluted Clorox (two parts Clorox plus eight parts water) is effective in reducing the bacterial population on a seed surface.','Leaf:Pepper bell\nCondition:healthy',
            'Leaf:Potato leaf-->Disease:Early blight-->Suggestions:Albina is our highly efficient calcium fertilizer which improves shelf life & firmness of potato tubers and reduces susceptibility to disease & physiological disorders', 
            'Leaf:Potato-->Condition:healthy',
            'Leaf:Tomato-->Disease:Bacterial spot-->Suggestion:Make up a solution of about a teaspoon of Epsom salts per litre (quarter gallon) of water in a spray bottle. Simply wet the foliage on your tomato plants every two weeks using a fine spray setting. It will quickly be absorbed by the leaves. Avoid spraying on hot, sunny days or when rain is imminent' ,
            'Leaf:Tomato-->Disease:Early blight-->Suggestion:Use protectant fungicides, mancozeb, chlorothalonil or copper products. Systemic products are available, e.g., strobilurins, although they are expensive and, if used too often, the fungus may develop resistance to them.', 
            'Leaf:Tomato-->Disease:Late blight-->Suggestion:Use fungicide sprays based on mandipropamid, chlorothalonil, fluazinam, mancozeb to combat late blight. Fungicides are generally needed only if the disease appears during a time of year when rain is likely or overhead irrigation is practiced.',
            'Leaf:Tomato-->Disease:Leaf Mold-->Suggestion:Active ingredient chlorothalonil is the most recommended chemical for us on tomato fungus. It can be applied until the day before you pick tomatoes, which is a clear indication of its low toxicity.' ,
            'Leaf:Tomato-->Disease:Septoria leaf spot-->Suggestion:Most fungicides registered for use on tomatoes would effectively control Septoria leaf spot. These include maneb, mancozeb, chlorothalonil, and benomyl. Captan is not effective and zineb may be difficult to purchase.',
            'Leaf:Tomato-->Disease:Two spotted spider mite-->Suggestion:Apply a miticide spray if mite damage and mites are present on 50 per cent of the leaves and predators are not present.This will effectively control the spread' ,
            'Leaf:Tomato-->Disease:Target Spot-->Suggestion:The effective agents are chlorothalonil, copper oxychloride or mancozeb. Treatment should start when the first spots are seen and continue at 10-14-day intervals',
            'Leaf:Tomato-->Disease:Yellow Leaf Curl Virus-->Suggestion:Imidacloprid should be sprayed on the entire plant and below the leaves; eggs and flies are often found below the leaves. Spray every 14-21 days and rotate on a monthly basis with Abamectin so that the whiteflies do not build-up resistance to chemicals.', 
            'Leaf:Tomato-->Disease:Mosaic virus-->Suggestion:To avoid seed-borne mosaic viruses, soak seeds of susceptible plants in a 10% bleach solution before planting.',
            'Leaf:Tomato-->Condition:healthy']
        return CATEGORIES[pred_class]

        return CATEGORIES[pred_class]
    return None


if __name__ == '__main__':
    app.run(debug=False)

