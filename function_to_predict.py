from keras.models import model_from_json
from keras.models import load_model


def output(img_path):
    img = image.load_img(img_path, target_size=(256,256,3), grayscale=False )
    img = image.img_to_array(img)
    img = img/255
    json_file = open('model_in_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_keras.h5")
    y_pred = model.predict_classes(img, batch_size=128, verbose=0)
    return y_pred