import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
from torchvision import models
import io

def is_image_file(filename):
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in ('.jpg', '.jpeg')

def process_image(image):
    img = image.resize((256, 256))
    width, height = 256, 256
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    img = np.array(img).transpose((2, 0, 1)) / 256
    img = img[:3,:,:]
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img - means) / stds
    img_tensor = torch.Tensor(img)
    return img_tensor

def model_loading():
    path = 'D:\\project\\medical herbs\\model\\resnet50-transfer.pth'
    model_name = os.path.basename(path).split('-')[0]if '-' in os.path.basename(path) else os.path.basename(path).split('.')[0]
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = checkpoint['fc']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    class_labels = [
    'Alpinia Galanga (Rasna)',
    'Amaranthus Viridis (Arive-Dantu)',
    'Artocarpus Heterophyllus (Jackfruit)',
    'Azadirachta Indica (Neem)',
    'Basella Alba (Basale)',
    'Brassica Juncea (Indian Mustard)',
    'Carissa Carandas (Karanda)',
    'Citrus Limon (Lemon)',
    'Ficus Auriculata (Roxburgh fig)',
    'Ficus Religiosa (Peepal Tree)',
    'Hibiscus Rosa-sinensis',
    'Jasminum (Jasmine)',
    'Mangifera Indica (Mango)',
    'Mentha (Mint)',
    'Moringa Oleifera (Drumstick)',
    'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    'Murraya Koenigii (Curry)',
    'Nerium Oleander (Oleander)',
    'Nyctanthes Arbor-tristis (Parijata)',
    'Ocimum Tenuiflorum (Tulsi)',
    'Piper Betle (Betel)',
    'Plectranthus Amboinicus (Mexican Mint)',
    'Pongamia Pinnata (Indian Beech)',
    'Psidium Guajava (Guava)',
    'Punica Granatum (Pomegranate)',
    'Santalum Album (Sandalwood)',
    'Syzygium Cumini (Jamun)',
    'Syzygium Jambos (Rose Apple)',
    'Tabernaemontana Divaricata (Crape Jasmine)',
    'Trigonella Foenum-graecum (Fenugreek)'
    ]

    model.class_to_idx = [(label, idx) for idx, label in enumerate(class_labels)]
    model.idx_to_class = [(idx, label) for idx, label in enumerate(class_labels)]

    return model

def predict(image, model, topk):
    img_tensor = process_image(image)
    img_tensor = img_tensor.reshape(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(topk, dim=1)
        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return top_p, top_classes

def extract(index):
    if index < 30:
        df = pd.read_csv("info1.csv")
        info = df.iloc[index, 2]
        return info
    else:
        return 'Not a valid Index'

def main():
    st.title("Herbs Detection App")

    uploaded_file = st.file_uploader("Choose a herb image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            st.write("Classifying...")
            image = Image.open(uploaded_file)
            model = model_loading()
            p, classes = predict(image, model, 1)
            info = extract(classes[0][0])
            data = {"prediction": classes[0][1], "confidence_level": p[0]*100, "info": info}
            st.write(f"Predicted Herb: {classes[0][1]}")
            st.write(f"Confidence Level: {p[0]*100:.2f}%")
            st.write(f"Additional Info: {info}")

if __name__ == "__main__":
    main()
