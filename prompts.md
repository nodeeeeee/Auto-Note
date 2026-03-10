## Context
The overview of this project can be seen in `task.md`
When developing this project, you should create a conda environment.

## Add canvas lecture video download feature
### File
`video_downloader.py`
### Task
This python program is used to download canvas lecture videos automatically and arrange the videos into course folders. I have provided you with canvas_token and canvas_credentials. 
You need to first login to my canvas, and list all the videos that can be downloaded. These videos should be downloaded into course folders ([project_folder]/[course_id]/videos)created by the program. 
The downloading step should be conducted by a tampermonkey script: Panopto-Video-DL.
### Finish Condition
You need to test your program step-by-step.
1. Succeed in listing all videos available on canvas. 
2. Succeed in downloading one of the videos from canvas.
3. Succeed in downloading five videos from canvas, and arrange them correctly in the local file system


## Build preprocessing pipeline
### File
`extract_caption.py`
### Task
We currently are able to download videos. You need to build a pipeline so that for each video downloaded, the program first use audio-separator package to separate human voice. And then use whipser model Large-V3 to generate the transcript. This should enable GPU if the user has the device. 
Also, make sure that video_downloader and extract_caption will not redo the work on existing files.

### Finish Condition
The program successfully extract caption for the videos and store them in proper directories.


## Add Download lecture files feature
### File
`material downloader`
### Task
You need to download the course materials of each course into course folders([project_folder]/[course_id]/materials).
You need to first calculate the estimated size of the course materials. If the total size is smaller than 1GB, then download all materials. Otherwise, you use AI to identify lecture notes and tutorials, and download only these two types of files.
The materials should be arranged properly inside the folders.
Keep a log, so that if a file is downloaded, do not download again. The log is used to synchronize the downloading.
### Finish Condition
Select three courses. Verify that the program successfully downloads all the files.

## Semantic Alignment (The "Agent" Logic)

### File
`semantic_alignment.py`

### Task
Input: caption, slides

Contextual Mapping: Build a mapping engine that uses the Whisper timestamps and Slide OCR data. The alignment should be based on keyword-slideOCR matching.

Heuristic: If the professor says "As you can see on this graph of the sigmoid function," the Agent must locate the slide containing "sigmoid" and "graph" and mark that timestamp as the "active period" for that slide.

You need to apply RAG because the files are too big to be stored in the context.

The slides might be pdf,pptx, or even word. Please provide functions for all of them.

You may use openai api (gpt4o) to enable the function. The api key is provided in `openai_api.txt`. However, you are not encouraged to do so. If using local models, you can perform very high accuracy, then don't use openai_api. 

### Finish Condition
If the program is able to generate a representation for sentence-slide matching. Store it in course folder ([project_folder]/[course_id]/alignment)


## Refine Semantic Alignment
### File
`semantic_alignment.py`

### Task
Currently we can only use text in the slides, but I want you to also include text in comment scripts. In addition, if a slide has image, the program should also extract data from the image for more precise positioning.
And you need to consider situations where the lecturer is talking about things out of the slide. You should also note this down.

### Finish Condition
By verifying it yourself, you need to make sure that the alignment has very high accuracy. If not, you might tune the model or use other techniques until you achieve the best results.
