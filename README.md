# FaceVerificationApp

FaceVerificationApp is a face verification application developed using Kivy and Siamese Neural Networks. The app allows users to authenticate their identity by comparing their face with a set of pre-registered face images. It provides a user-friendly interface, real-time webcam integration, and utilizes deep learning techniques for accurate verification.

## Features

- Real-time webcam feed for capturing face images.
- Preprocessing of captured images for compatibility with the model.
- Siamese Neural Network architecture for calculating similarity scores.
- Adjustable detection and verification thresholds for customizable verification criteria.
- User-friendly GUI developed with Kivy framework.
- Integration with TensorFlow and OpenCV libraries.

## How it Works

1. The app captures an image of the user's face using the webcam.
2. The captured image is preprocessed to match the input requirements of the Siamese Neural Network model.
3. The model compares the input image with a set of pre-registered face images.
4. Similarity scores are calculated for each comparison.
5. The scores are compared against detection and verification thresholds.
6. The app displays the verification status as "Verified" or "Un-verified" based on the outcome.

## Implementation Details

This project is an implementation of the paper titled "Siamese Neural Networks for One-shot Image Recognition." The Siamese Neural Network architecture described in the paper is utilized to perform the face verification task. The model is trained using a dataset that includes images of the user's face as well as images of other individuals from the "Labelled Faces in the Wild" dataset.

## Dependencies

- Kivy: Python framework for developing multi-touch applications.
- TensorFlow: Deep learning library for building and training neural networks.
- OpenCV: Computer vision library for image and video processing.
- NumPy: Library for numerical computing and array operations.

## Usage

1. Clone the repository: `git clone [repository_url]`
2. Open the Jupyter Notebook `faceid.ipynb`.
3. Run the notebook. It will automatically install the required dependencies and create directories to save the training and testing datasets in your local folder.
4. Follow the instructions in the notebook to capture images of yourself for training the model.
5. Once the training is complete, run the `main.py` script to launch the face verification application.
6. The application will utilize the trained model to verify your face against the registered face images.

By following these steps, you can easily set up the project, capture your own face images, train the model, and then use the face verification app for authentication.

## Credits

This project is based on the tutorial video created by Nicholas Renotte on YouTube. I would like to acknowledge and give credit to Nicholas Renotte for his valuable contribution to the development of this project.

## Conclusion

FaceVerificationApp demonstrates the potential of deep learning and computer vision techniques in building real-world applications. By leveraging the Siamese Neural Network architecture and the Kivy framework, this project provides a reliable and user-friendly face verification solution. I hope you find this project insightful and inspiring for your own endeavors in deep learning and application development.


Feel free to explore the code, contribute to the project, and utilize it for your own applications.

#FacialRecognition #DeepLearning #Kivy #SiameseNeuralNetworks #OpenCV
