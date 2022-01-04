# Video understanding

App developed using [Gradio] to test video understading tools. Currently running [PyTorchVideo] as prediction tool.

### Current Status ###
<p align="center">
  <img src="https://github.com/agarciagoni/video_understanding/blob/main/examples/status_huggingface.PNG" width="900" height="500">
  </p>

### Install ###
To install requirements needed
```bash
$ pip install -r requirements.txt
```


[Gradio]:https://gradio.app/
[PyTorchVideo]:https://pytorchvideo.org/ 

### To run the tool ###
Get the repository with labels and an example video.
Two main files with webcam or file input.

```bash
$ git clone https://github.com/agarciagoni/video_understanding
$ cd video_understanding
$ python main_webcam.py #webcam input
$ python main_file_input.py #video_file_input
```
