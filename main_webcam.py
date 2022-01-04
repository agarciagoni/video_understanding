import gradio as gr
import functions as f


#build interface

#inputs
num_frames = gr.inputs.Slider(minimum=10, maximum=64,step = 2, label="Number of frames:", default=32)
frames_per_second = gr.inputs.Slider(minimum=15, maximum=60,step = 15,label="Frames per second:", default=30)
video = gr.inputs.Video(type=None, source="webcam", label='Video input', optional=False)

#ouputs
video_length = gr.outputs.Textbox(label="Video length analyzed (seconds) :")
prediction = gr.outputs.Label(num_top_classes=5, label="Top 5 Predictions:")

iface = gr.Interface(
    f.video_predict,
    inputs= [num_frames, frames_per_second, video],
    outputs = [video_length,prediction],
    title = 'Video scene predictor',
    description='Test to predict the scene of a video',
#    examples='C:/Users/alejandro.garcia/Desktop/VASS/Video Analysis/archery.mp4'
    css = 'base.css',
    theme = 'darkhuggingface'
)
iface.launch()