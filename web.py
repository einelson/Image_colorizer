"""
Author: Ethan Nelson
File: collage_generator.py
Description: 
    Using OpenCV to constuct a collage.
    Border color, rotation and placement are randomly generated

"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
import numpy as np
from cv2 import cv2
import base64
from itertools import chain
import plotly.express as px

# global
app = dash.Dash(__name__, suppress_callback_exceptions=True,update_title=None, title='Ethan Nelson', external_stylesheets=[dbc.themes.DARKLY, 'website.css'])
server = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


images_colorization = []
image_names_colorization = []
# model = '.' + app.get_asset_url('data/image_colorization/model0.h5')

'''
LAYOUT
'''
app.layout = html.Div([
    
    # row0
    dbc.Row(
        [
            # about
            dbc.Col(
                [
                    html.H1('Image colorization', className='title'),
                    html.P('The image colorization project was based around seeing if you could predict the colored counterpart of a greyscale image.', className='paragraph'),
                    html.P('The model was based off of learning on a broad set of images and the goal was to predict image with any amalgam of colors.', className='paragraph'),
                    html.P('The model built was not complex enough or trained enough to predict colorful images well and this particular model loves to predict reds, blues and oranges.', className='paragraph'),
                    html.P('In order to fine tune this in the future I would make the image prediction less generic and be used to predict only like-images: images that come from the same source and have the saemfinal format.', className='paragraph'),                    
                    html.P('*Click on the image to the right to link to this projects github.', className='paragraph'),
                ],
            ),
            # picture
            dbc.Col(
                [
                    html.A([html.Img(src=app.get_asset_url('/data/image_colorization/frog.png'), className='image')], href='https://github.com/einelson/Image_colorizer'),
                ],
            ),            
        ],
    ),

    # row1
    dbc.Row(
        [
            # picture
            dbc.Col(
                [
                    html.Img(src=app.get_asset_url('/data/image_colorization/image_colorization_lizard.png'), className='image'),
                ],
            ),
            # about
            dbc.Col(
                [
                    html.H2('Use Cases', className='title'),
                    html.P('Why use image colorization, why not just use colored images?', className='paragraph'),
                    html.P('Everything takes up some sort of space. Images are generally made up of 3 channels consisting of R(red), G(green), B(blue). When sending or storing data you need to save the 3 channels that the image consists of.', className='paragraph'),
                    html.P('Lets say you want to send or store a large image in an enviroment where there is restrictions in space, or speed in sending the image. By converting the image from RGB to a L(perceptual lightness) A(red, green) B(blue, yellow) colorscale and only using the L channel you are effectivley saving 3x the space. The L channel is considered greyscale ond only takes up one channel.', className='paragraph'),
                    html.P('Using image colorization we recreate the exact image when it is needed. While the algorithm I built as an example for this is not close to production worthy it is a working example of how this is an achievable goal.', className='paragraph'),
                ],
            ),            
        ],style={'margin-top': '4em'},
    ),

    # row2
    dbc.Row(
        [
            # tools
            dbc.Col(
                [
                    dbc.Card(
                        [   
                            # header
                            dbc.CardHeader('Upload Image'),
                            # settings
                            dbc.CardBody(
                                [
                                    html.P('Please upload only one image. Images will be rescaled to 256x256.'),
                                    # upload images
                                    html.P('Upload images', className='input_line'),
                                    html.Div(className='images_list'),
                                    html.Div(id='output-image-upload-colorization'),
                                    dcc.Upload(
                                        id='upload-image-colorization',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Files')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=False
                                    ),

                                    # clear uploaded
                                    dbc.Button('Clear', color='secondary', id='b_clear_uploaded_colorization'),
                                ],
                            ),

                            # execute
                            dbc.CardFooter(
                                [
                                    dbc.Button('Predict', color='success', id='b_predict_colorization'),
                                ],
                            ),
                        ],style={'margin-top': '6em'},
                    ),
                ],
            ),
            # picture output
            dbc.Col(
                [
                    dcc.Graph(id='out_predict_colorization', className='image_gen'),
                ],
            ),
        ],style={'margin-top': '4em'},
    ),
], className='layout')


'''
CALLBACKS
'''
# uploaded images
def read_image_string(contents):
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@app.callback(
    Output('output-image-upload-colorization', 'children'),
    Input('b_clear_uploaded_colorization', 'n_clicks'),
    Input('upload-image-colorization', 'contents'),
    State('upload-image-colorization', 'filename'),
    State('upload-image-colorization', 'last_modified'))
def update_output(n, list_of_contents, list_of_names, list_of_dates):
    triggered = [p['prop_id'].split('.')[0] for p in dash.callback_context.triggered]
    
    if 'upload-image-colorization' in triggered:
        global images_colorization
        images_colorization = []
        images_colorization.append(read_image_string(list_of_contents))
        global image_names_colorization
        image_names_colorization = []
        image_names_colorization.append(list_of_names)
        return list(chain.from_iterable(image_names_colorization))
    
    else: # clear images on new page or refresh or clear
        images_colorization = []
        image_names_colorization = []


# on execute
@app.callback(
    Output('out_predict_colorization', 'figure'),
    Input('b_predict_colorization', 'n_clicks'),
)
def predict_colorization(n):
    triggered = [p['prop_id'].split('.')[0] for p in dash.callback_context.triggered]
    if 'b_predict_colorization' in triggered:
        global images_colorization
        # make sure everything is filled out
        try:
            collage = px.imshow(main(images_colorization), title='Image Colorization')
            collage.update_layout(paper_bgcolor = 'rgb(34, 34, 34)', font = {'color': "white", 'family': "Arial"})
            return collage
        except:
            img = np.arange(100).reshape((10, 10))
            fig = px.imshow(img, binary_string=True, title='Image Colorization')
            fig.update_layout(paper_bgcolor = 'rgb(34, 34, 34)', font = {'color': "white", 'family': "Arial"})
            return fig            
    else:
        img = np.arange(100).reshape((10, 10))
        fig = px.imshow(img, binary_string=True, title='Image Colorization')
        fig.update_layout(paper_bgcolor = 'rgb(34, 34, 34)', font = {'color': "white", 'family': "Arial"})
        return fig


'''
prediction
'''

def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

def main(image):
    # Converts RGB values to LAB
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('.' + app.get_asset_url('data/image_colorization/model0.h5'))
    
    # resize
    image[0] = cv2.resize(image[0], (256,256), fx=1, fy=1)
    
    #Load test images
    x = get_lab(image[0])

    # prepare for network
    test_image_x = np.zeros((256,256,1))
    test_image_x[:,:,0] = x
    test_image = np.stack([test_image_x, test_image_x])
    # print(f'test imagex shape{test_image_x.shape}')

    # test network on an image
    output = (model.predict(test_image)[0]) *128
    # print(f'output image shape {output.shape}')
    cur = np.zeros((256,256,3))
    cur[:,:,0] = test_image_x[:,:,0] # L layer
    cur[:,:,1:] = output # A B layers
    rgb_image = lab2rgb(cur)

    return rgb_image


if __name__ == '__main__':
    app.run_server(debug=False, port=8000)