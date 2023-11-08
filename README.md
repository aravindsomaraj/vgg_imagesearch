# **Image Search Engine using Python and Jupyter**
---

This project is a result of a course on Deep Learning and Convoluted Nueral Networks (CNN), where we build an image search engine using open source Python and Jupyter framework. This project includes a Jupyter notebook that uses VGG models of the CNN architecture to extract image embeddings and text embeddings and a Flask web application that allows users to search for specific images.

## Features
- **Image and text embeddings**: In addition to searching for similar images, the search engine can also perform *text to image* and *image to text* searches using image and text embeddings. This means that users can input text or an image and the search engine will return the most relevant images or text based on their similarities to the input.
- **Approximate nearest neighbor algorithm**: To speed up the search process, the search engine uses an approximate nearest neighbor algorithm. This algorithm finds the nearest neighbors to a given image by searching through a smaller set of representative vectors rather than the entire dataset.
- **Fine-tuning accuracy of the model**: The search engine allows for fine-tuning the accuracy of the model by adjusting the parameters and training the model on a smaller dataset. <!--This means that users can optimize the model to work for their specific use case and improve its accuracy.-->
- **Real-life web application**: The search engine comes with a real-life web application that allows users to perform visual similarity searches by uploading their own images or using pre-existing ones. <!--The web application provides an easy-to-use interface for users to interact with the search engine and get relevant results.-->
## Installation
Follow the instructions below:  
1. Clone the repository
   ```
   git clone https://github.com/aravindsomaraj/vgg_imagesearch.git
   ```
2. Install the required packages
    ```
    pip install -r requirements.txt
    ```

## How to run?
- To run the Jupyter notebook, open terminal in the project directory and run the following:
  ```
  jupyter notebook
  ```
- Navigate to **'imagesearch.ipynb'** file in the Jupyter notebook opened in the web browser and run the cells one by one.
- To run the web application, open the terminal in the project directory and run the following commands:
  ```
  cd imagesearch
  python imagesearch.py
  ```
  This will start the Flask server and you can access the web application in your browser in your localhost server.
