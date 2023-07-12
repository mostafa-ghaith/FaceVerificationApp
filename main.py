# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app layout
class CamApp(App):

    def build(self):
        # Creating main components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press =self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Adding compnents to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        # Load trained tensorflow model
        self.model = tf.keras.models.load_model('siamesemodel.h5',custom_objects={'L1Dist':L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continouosly to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]    

        # Flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocess images before feeding into the model
    def preprocess(self, file_path):
    
        # Read image from file path
        byte_img = tf.io.read_file(file_path)
        # Load image in jpeg format
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing
        # Resizing image to 100x100x3
        img = tf.image.resize(img, (105,105))
        # Scale image between 0 and 1
        img = img/255.0
        
        return img
    
    # Verify image
    def verify(self, *args):
        # specify thresholds
        detection_threshold = 0.8
        verification_threshold = 0.8

        # Save input image from webcam
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_image = self.preprocess(os.path.join('application_data','verification_images',image))
            
            # The model expects input data in a batch format, even for a single image.
            # np.expand_dims() adds an extra dimension to the input data, creating a batch with a single image.
            # list() converts the resulting array into a Python list, which can be passed to the model's predict() function.
            result = self.model.predict(list(np.expand_dims([input_img, validation_image], axis = 1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        # Update verification text
        self.verification_label.text = 'verified' if verified==True else 'Un-verified'

        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.5))

        return results, verified


if __name__ == '__main__':
    CamApp().run()
