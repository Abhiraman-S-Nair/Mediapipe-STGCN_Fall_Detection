---
jupyter:
  accelerator: GPU
  colab:
    gpuType: T4
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="pbEU4rWeBR3w"}
#Installing Libraries
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="vgW8vOdnCzop" outputId="b457624c-3f0f-4264-8396-bf1abffba4c3"}
``` python
!pip install ultralytics torch torchvision opencv-python mediapipe numpy matplotlib
```

::: {.output .stream .stdout}
    Collecting ultralytics
      Downloading ultralytics-8.3.50-py3-none-any.whl.metadata (35 kB)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.5.1+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.20.1+cu121)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.10.0.84)
    Collecting mediapipe
      Downloading mediapipe-0.10.20-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (9.7 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (1.26.4)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)
    Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (11.0.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (6.0.2)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.32.3)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.13.1)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.66.6)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from ultralytics) (5.9.5)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.0.0)
    Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.2.2)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.13.2)
    Collecting ultralytics-thop>=2.0.0 (from ultralytics)
      Downloading ultralytics_thop-2.0.13-py3-none-any.whl.metadata (9.4 kB)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.16.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2024.10.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from mediapipe) (1.4.0)
    Requirement already satisfied: attrs>=19.1.0 in /usr/local/lib/python3.10/dist-packages (from mediapipe) (24.2.0)
    Requirement already satisfied: flatbuffers>=2.0 in /usr/local/lib/python3.10/dist-packages (from mediapipe) (24.3.25)
    Requirement already satisfied: jax in /usr/local/lib/python3.10/dist-packages (from mediapipe) (0.4.33)
    Requirement already satisfied: jaxlib in /usr/local/lib/python3.10/dist-packages (from mediapipe) (0.4.33)
    Requirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.10/dist-packages (from mediapipe) (4.10.0.84)
    Requirement already satisfied: protobuf<5,>=4.25.3 in /usr/local/lib/python3.10/dist-packages (from mediapipe) (4.25.5)
    Collecting sounddevice>=0.4.4 (from mediapipe)
      Downloading sounddevice-0.5.1-py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.10/dist-packages (from mediapipe) (0.2.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2024.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2024.8.30)
    Requirement already satisfied: CFFI>=1.0 in /usr/local/lib/python3.10/dist-packages (from sounddevice>=0.4.4->mediapipe) (1.17.1)
    Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax->mediapipe) (0.4.1)
    Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax->mediapipe) (3.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from CFFI>=1.0->sounddevice>=0.4.4->mediapipe) (2.22)
    Downloading ultralytics-8.3.50-py3-none-any.whl (898 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 899.0/899.0 kB 37.3 MB/s eta 0:00:00
    ediapipe-0.10.20-cp310-cp310-manylinux_2_28_x86_64.whl (35.6 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.6/35.6 MB 54.2 MB/s eta 0:00:00
    ediapipe
    Successfully installed mediapipe-0.10.20 sounddevice-0.5.1 ultralytics-8.3.50 ultralytics-thop-2.0.13
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="JiQYep2xmyoN" outputId="52333c32-5946-4b64-c293-4247cb904c10"}
``` python
pip install tqdm
```

::: {.output .stream .stdout}
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (4.66.6)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="fbPmC1r_igpH" outputId="7a55ad20-4786-4ca2-ab78-1d124e727382"}
``` python
pip install twilio
```

::: {.output .stream .stdout}
    Collecting twilio
      Downloading twilio-9.4.1-py2.py3-none-any.whl.metadata (12 kB)
    Requirement already satisfied: requests>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from twilio) (2.32.3)
    Requirement already satisfied: PyJWT<3.0.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from twilio) (2.10.1)
    Requirement already satisfied: aiohttp>=3.8.4 in /usr/local/lib/python3.10/dist-packages (from twilio) (3.11.10)
    Collecting aiohttp-retry==2.8.3 (from twilio)
      Downloading aiohttp_retry-2.8.3-py3-none-any.whl.metadata (8.9 kB)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (2.4.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (1.3.1)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (4.0.3)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (0.2.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp>=3.8.4->twilio) (1.18.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.0.0->twilio) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.0.0->twilio) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.0.0->twilio) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.0.0->twilio) (2024.8.30)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from multidict<7.0,>=4.5->aiohttp>=3.8.4->twilio) (4.12.2)
    Downloading twilio-9.4.1-py2.py3-none-any.whl (1.9 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 49.6 MB/s eta 0:00:00
:::
:::

::: {.cell .markdown id="hF_j03yoA7Je"}
#YOLO Model Setup
:::

::: {.cell .markdown id="FW3JzV5kA_yG"}
##Function
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="0j1MnQKuA60l" outputId="b6e6d50c-4599-43e7-85cc-117d3d1a1755"}
``` python
from ultralytics import YOLO
import cv2
import os

# Initialize YOLO model
model = YOLO('yolov8n.pt')

def detect_bounding_boxes(video_path, output_folder):
    """
    Detects bounding boxes of people in a video and saves cropped frames.
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save cropped frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # Save frames with bounding boxes
            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == 0 and conf > 0.5:  # Class 0 is for 'person'
                    x1, y1, x2, y2 = map(int, box)
                    cropped_person = frame[y1:y2, x1:x2]
                    cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", cropped_person)

        frame_count += 1

    cap.release()
    print(f"Bounding boxes detected and saved in {output_folder}")
```

::: {.output .stream .stdout}
    Creating new Ultralytics Settings v0.0.6 file ✅ 
    View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
    Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
    Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt'...
:::

::: {.output .stream .stderr}
    100%|██████████| 6.25M/6.25M [00:00<00:00, 254MB/s]
:::
:::

::: {.cell .markdown id="544zD-vlBB81"}
##Execution
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":377}" collapsed="true" id="djB6_vP9AyJ2" outputId="ad6ed363-aaac-46ff-a644-3727caf9c225"}
``` python
from google.colab import files

# Upload video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
output_folder = '/content/cropped_frames'

# Detect bounding boxes and save frames
detect_bounding_boxes(video_path, output_folder)
```

::: {.output .display_data}
```{=html}

     <input type="file" id="files-1f9644de-6c06-4c7a-85da-a569f5d1ea41" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-1f9644de-6c06-4c7a-85da-a569f5d1ea41">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 
```
:::

::: {.output .stream .stdout}
    Saving person_walking.mp4 to person_walking.mp4
:::

::: {.output .error ename="AttributeError" evalue="'numpy.ndarray' object has no attribute 'dim'"}
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-46-ab03cf214ac0> in <cell line: 9>()
          7 
          8 # Detect bounding boxes and save frames
    ----> 9 detect_bounding_boxes(video_path, output_folder)

    <ipython-input-5-2165cccfde32> in detect_bounding_boxes(video_path, output_folder)
         25 
         26         # Perform detection
    ---> 27         results = model(frame)
         28         for result in results:
         29             boxes = result.boxes.xyxy.cpu().numpy()

    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1734             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1735         else:
    -> 1736             return self._call_impl(*args, **kwargs)
       1737 
       1738     # torchrec tests the code consistency with the following code

    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1745                 or _global_backward_pre_hooks or _global_backward_hooks
       1746                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747             return forward_call(*args, **kwargs)
       1748 
       1749         result = None

    <ipython-input-11-fe60d1ee7c7c> in forward(self, x)
         27 
         28     def forward(self, x):
    ---> 29         if x.dim() == 3:
         30             x = x.unsqueeze(1)  # Shape becomes (batch_size, 1, num_keypoints, num_frames)
         31 

    AttributeError: 'numpy.ndarray' object has no attribute 'dim'
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="esp1j57XWHXh" outputId="2fc254cc-6a57-4483-bdc9-b13a04de2d3b"}
``` python
print(video_path)
```

::: {.output .stream .stdout}
    person_walking.mp4
:::
:::

::: {.cell .markdown id="PKpO5nJeDla0"}
#Pose Keypoint Extraction
:::

::: {.cell .markdown id="Xs2tmX30Dtgr"}
##Function
:::

::: {.cell .code id="ZlDx_NwIDsnV"}
``` python
import mediapipe as mp
import numpy as np
import cv2
import os

def extract_keypoints_and_labels(image_folder, output_keypoints_path, output_labels_path):
    """
    Extracts pose keypoints and labels from images.
    Args:
        image_folder (str): Path to the folder containing images.
        output_keypoints_path (str): Path to save the keypoints numpy file.
        output_labels_path (str): Path to save the labels numpy file.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    keypoints_list = []
    labels_list = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            label = 1 if 'fallen' in image_name.lower() else 0

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = [
                    [lm.x, lm.y, lm.z]
                    for lm in results.pose_landmarks.landmark
                ]
            else:
                # If no pose is detected, add zeros
                keypoints = [[0, 0, 0] for _ in range(33)]

            keypoints_list.append(keypoints)
            labels_list.append(label)

    pose.close()

    # Save keypoints and labels as numpy arrays
    keypoints_array = np.array(keypoints_list)
    labels_array = np.array(labels_list)
    np.save(output_keypoints_path, keypoints_array)
    np.save(output_labels_path, labels_array)
    print(f"Keypoints saved to {output_keypoints_path}")
    print(f"Labels saved to {output_labels_path}")
```
:::

::: {.cell .markdown id="kKDHiXSDDyIL"}
##Execution
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="roBcC3yRD0Sq" outputId="e71233c8-3307-4e7e-bdb4-11d717a7d141"}
``` python
# Mount Drive to load dataset
from google.colab import drive
drive.mount('/content/drive')

# Paths to your datasets
train_images_path = '/content/drive/MyDrive/STGCN_Dataset/images/train'
validation_images_path = '/content/drive/MyDrive/STGCN_Dataset/images/validation'

# Output paths for keypoints and labels
train_keypoints_path = '/content/drive/MyDrive/STGCN_Dataset/train_keypoints.npy'
train_labels_path = '/content/drive/MyDrive/STGCN_Dataset/train_labels.npy'
validation_keypoints_path = '/content/drive/MyDrive/STGCN_Dataset/validation_keypoints.npy'
validation_labels_path = '/content/drive/MyDrive/STGCN_Dataset/validation_labels.npy'

# Extract keypoints and labels for train and validation datasets
extract_keypoints_and_labels(train_images_path, train_keypoints_path, train_labels_path)
extract_keypoints_and_labels(validation_images_path, validation_keypoints_path, validation_labels_path)
```

::: {.output .stream .stdout}
    Mounted at /content/drive
    Keypoints saved to /content/drive/MyDrive/STGCN_Dataset/train_keypoints.npy
    Labels saved to /content/drive/MyDrive/STGCN_Dataset/train_labels.npy
    Keypoints saved to /content/drive/MyDrive/STGCN_Dataset/validation_keypoints.npy
    Labels saved to /content/drive/MyDrive/STGCN_Dataset/validation_labels.npy
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="z8sMojD0HcBq" outputId="0ef952b8-ede7-4668-a294-c426f77ab8e7"}
``` python
# Verify train keypoints and labels
train_keypoints = np.load(train_keypoints_path)
train_labels = np.load(train_labels_path)
print("Train Keypoints Shape:", train_keypoints.shape)
print("Train Labels Shape:", train_labels.shape)

# Verify validation keypoints and labels
validation_keypoints = np.load(validation_keypoints_path)
validation_labels = np.load(validation_labels_path)
print("Validation Keypoints Shape:", validation_keypoints.shape)
print("Validation Labels Shape:", validation_labels.shape)
```

::: {.output .stream .stdout}
    Train Keypoints Shape: (332, 33, 3)
    Train Labels Shape: (332,)
    Validation Keypoints Shape: (86, 33, 3)
    Validation Labels Shape: (86,)
:::
:::

::: {.cell .markdown id="ROGIc7ccD9Ez"}
#STGCN Model Definition
:::

::: {.cell .code id="sb4KhN7-KeXW"}
``` python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define the STGCN layer
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_spatial=25, kernel_size_temporal=3):
        super(STGCNLayer, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size_spatial), padding=(0, kernel_size_spatial // 2))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_temporal, 1), padding=(kernel_size_temporal // 2))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.spatial_conv(x)  # Spatial convolution
        x = torch.relu(x)
        x = self.temporal_conv(x)  # Temporal convolution
        x = self.bn(torch.relu(x))  # Batch normalization
        return x

# Define the full STGCN model
class STGCN(nn.Module):
    def __init__(self, num_keypoints, num_classes):
        super(STGCN, self).__init__()
        self.stgcn1 = STGCNLayer(1, 64, num_keypoints)  # Change input channels to 1
        self.stgcn2 = STGCNLayer(64, 64, num_keypoints)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape becomes (batch_size, 1, num_keypoints, num_frames)

        x = x.permute(0, 1, 3, 2)  # Reorder to (batch_size, in_channels, num_frames, num_keypoints)
        x = self.stgcn1(x)
        x = self.stgcn2(x)
        x = x.mean(dim=[2, 3])  # Global average pooling (across frames and keypoints)
        return self.fc(x)


# Dataset class
class PoseDataset(Dataset):
    def __init__(self, keypoints, labels):
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)  # Shape: (N, num_keypoints, 3)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.labels[idx]
```
:::

::: {.cell .markdown id="Zr2gwB-SESm7"}
#Training the STGCN Model
:::

::: {.cell .markdown id="wFxfzU1HtHij"}
##Function
:::

::: {.cell .code id="R0J-nXprHFkG"}
``` python
from tqdm import tqdm

def train_stgcn(train_keypoints, train_labels, val_keypoints, val_labels, model_path, num_epochs=10, batch_size=16):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to tensors, if not already done
    train_keypoints = torch.tensor(train_keypoints, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.int64).to(device)
    val_keypoints = torch.tensor(val_keypoints, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.int64).to(device)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_keypoints, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_keypoints, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model with correct input channels and output classes
    model = STGCN(num_keypoints=train_keypoints.shape[2], num_classes=2)  # Assuming 33 keypoints and 2 classes
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for keypoints, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            keypoints, labels = keypoints.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(keypoints)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints, labels = keypoints.to(device), labels.to(device)
                outputs = model(keypoints)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
```
:::

::: {.cell .markdown id="vgydo3nGK91G"}
##Execution
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4Pg4RiGLK_0y" outputId="a8f408b7-6b4d-4da2-ec70-878dfc8130bf"}
``` python
# Load dataset
train_keypoints = np.load('/content/drive/MyDrive/STGCN_Dataset/train_keypoints.npy')
train_labels = np.load('/content/drive/MyDrive/STGCN_Dataset/train_labels.npy')

validation_keypoints = np.load('/content/drive/MyDrive/STGCN_Dataset/validation_keypoints.npy')
validation_labels = np.load('/content/drive/MyDrive/STGCN_Dataset/validation_labels.npy')

# Reshape the keypoints for STGCN
train_keypoints = train_keypoints.transpose(0, 2, 1)  # Shape: (332, 33, 3)
validation_keypoints = validation_keypoints.transpose(0, 2, 1)  # Shape: (86, 33, 3)

# Ensure the shapes are correct
print(f'Train Keypoints shape: {train_keypoints.shape}')
print(f'Train Labels shape: {train_labels.shape}')
print(f'Validation Keypoints shape: {validation_keypoints.shape}')
print(f'Validation Labels shape: {validation_labels.shape}')

# Train STGCN model
model_path = '/content/drive/MyDrive/STGCN_Dataset/stgcn_model.pth'

#train_stgcn(train_keypoints, train_labels, validation_keypoints, validation_labels, model_path, num_epochs=10, batch_size=16)
```

::: {.output .stream .stdout}
    Train Keypoints shape: (332, 3, 33)
    Train Labels shape: (332,)
    Validation Keypoints shape: (86, 3, 33)
    Validation Labels shape: (86,)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Z9hVtTspsw5V" outputId="29e3ec44-97c7-4192-fb05-cee7e4ec9ac5"}
``` python
import torch

# Convert labels from numpy arrays to PyTorch tensors and ensure they are integers
train_labels = torch.tensor(train_labels).long()
val_labels = torch.tensor(validation_labels).long()

print(f"Train Keypoints shape: {train_keypoints.shape}, dtype: {train_keypoints.dtype}")
print(f"Train Labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
print(f"Validation Keypoints shape: {validation_keypoints.shape}, dtype: {validation_keypoints.dtype}")
print(f"Validation Labels shape: {val_labels.shape}, dtype: {val_labels.dtype}")
```

::: {.output .stream .stdout}
    Train Keypoints shape: (332, 3, 33), dtype: float64
    Train Labels shape: torch.Size([332]), dtype: torch.int64
    Validation Keypoints shape: (86, 3, 33), dtype: float64
    Validation Labels shape: torch.Size([86]), dtype: torch.int64
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="AVtij-n3rw3q" outputId="7e70490e-ce06-47c0-c077-fcc20fe3e26c"}
``` python
# Example inputs
'''train_keypoints = torch.rand(332, 3, 33)  # Shape: (num_samples, channels, num_keypoints)
train_labels = torch.randint(0, 2, (332,))  # Shape: (num_samples,)
val_keypoints = torch.rand(86, 3, 33)       # Shape: (num_samples, channels, num_keypoints)
val_labels = torch.randint(0, 2, (86,))     # Shape: (num_samples,)'''

model_path = '/content/drive/MyDrive/STGCN_Dataset/stgcn_model.pth'

train_stgcn(train_keypoints, train_labels, validation_keypoints, val_labels, model_path, num_epochs=10, batch_size=16)
```

::: {.output .stream .stderr}
    <ipython-input-12-90dbd9c70c19>:9: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      train_labels = torch.tensor(train_labels, dtype=torch.int64).to(device)
    <ipython-input-12-90dbd9c70c19>:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      val_labels = torch.tensor(val_labels, dtype=torch.int64).to(device)
    Training Epoch 1: 100%|██████████| 21/21 [00:00<00:00, 29.42it/s]
:::

::: {.output .stream .stdout}
    Epoch [1/10], Loss: 0.6128
    Validation Accuracy: 52.33%
:::

::: {.output .stream .stderr}
    Training Epoch 2: 100%|██████████| 21/21 [00:00<00:00, 163.54it/s]
:::

::: {.output .stream .stdout}
    Epoch [2/10], Loss: 0.4866
    Validation Accuracy: 76.74%
:::

::: {.output .stream .stderr}
    Training Epoch 3: 100%|██████████| 21/21 [00:00<00:00, 200.41it/s]
:::

::: {.output .stream .stdout}
    Epoch [3/10], Loss: 0.4341
    Validation Accuracy: 59.30%
:::

::: {.output .stream .stderr}
    Training Epoch 4: 100%|██████████| 21/21 [00:00<00:00, 204.77it/s]
:::

::: {.output .stream .stdout}
    Epoch [4/10], Loss: 0.4128
    Validation Accuracy: 59.30%
:::

::: {.output .stream .stderr}
    Training Epoch 5: 100%|██████████| 21/21 [00:00<00:00, 202.45it/s]
:::

::: {.output .stream .stdout}
    Epoch [5/10], Loss: 0.3793
    Validation Accuracy: 65.12%
:::

::: {.output .stream .stderr}
    Training Epoch 6: 100%|██████████| 21/21 [00:00<00:00, 200.47it/s]
:::

::: {.output .stream .stdout}
    Epoch [6/10], Loss: 0.3951
    Validation Accuracy: 86.05%
:::

::: {.output .stream .stderr}
    Training Epoch 7: 100%|██████████| 21/21 [00:00<00:00, 206.55it/s]
:::

::: {.output .stream .stdout}
    Epoch [7/10], Loss: 0.3708
    Validation Accuracy: 80.23%
:::

::: {.output .stream .stderr}
    Training Epoch 8: 100%|██████████| 21/21 [00:00<00:00, 195.45it/s]
:::

::: {.output .stream .stdout}
    Epoch [8/10], Loss: 0.3615
    Validation Accuracy: 79.07%
:::

::: {.output .stream .stderr}
    Training Epoch 9: 100%|██████████| 21/21 [00:00<00:00, 181.61it/s]
:::

::: {.output .stream .stdout}
    Epoch [9/10], Loss: 0.3606
    Validation Accuracy: 81.40%
:::

::: {.output .stream .stderr}
    Training Epoch 10: 100%|██████████| 21/21 [00:00<00:00, 213.26it/s]
:::

::: {.output .stream .stdout}
    Epoch [10/10], Loss: 0.3509
    Validation Accuracy: 82.56%
:::
:::

::: {.cell .code id="rUJUi7STJeQ9"}
``` python
model_path = '/content/drive/MyDrive/STGCN_Dataset/stgcn_model.pth'
torch.save(model.state_dict(), model_path)
```
:::

::: {.cell .markdown id="K2tKXanfJzsC"}
#Prediction
:::

::: {.cell .markdown id="cbf3fUmqsgkQ"}
##Function
:::

::: {.cell .code id="46rZ9l84uDQm"}
``` python
import torch
import cv2
import numpy as np

def predict_fall_in_video(video_path, model, device, pose_extractor, frame_skip=50):
    """
    Predict whether a fall occurs in a given video using a trained ST-GCN model.

    Parameters:
        video_path (str): Path to the input video.
        model (torch.nn.Module): Trained ST-GCN model.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        pose_extractor (function): Function to extract pose keypoints from video frames.
        frame_skip (int): Number of frames to skip during processing to reduce RAM usage.

    Returns:
        dict: Dictionary with per-frame predictions and overall result.
    """
    model.eval()  # Set model to evaluation mode

    # Step 1: Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:  # Process every `frame_skip` frame
            frames.append(frame)
        frame_count += 1
    cap.release()

    # Step 2: Extract pose keypoints from the selected frames
    keypoints_sequence = []
    for frame in frames:
        keypoints = pose_extractor(frame)  # Extract keypoints for the frame
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
        else:
            keypoints_sequence.append(np.zeros((33, 4)))  # Handle missing frames

    # Step 3: Convert keypoints to tensor and reshape for ST-GCN
    keypoints_sequence = np.array(keypoints_sequence)  # Shape: (T, J, C)
    keypoints_sequence = np.expand_dims(keypoints_sequence, axis=0)  # Add batch dimension

    # Optionally collapse the keypoints to a single channel, e.g., by averaging x, y, z, visibility
    keypoints_sequence = keypoints_sequence.mean(axis=-1, keepdims=True)  # Average across the last axis (x, y, z, visibility)

    # Convert to tensor and move to the appropriate device
    keypoints_tensor = torch.tensor(keypoints_sequence, dtype=torch.float32).to(device)

    # Ensure the tensor is of the correct shape: [batch_size, channels, height, width]
    keypoints_tensor = keypoints_tensor.permute(0, 3, 1, 2)  # Reorder to [batch_size, channels, height, width]

    # Step 4: Pass keypoints through the ST-GCN model
    with torch.no_grad():
        outputs = model(keypoints_tensor)  # Model inference
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # Get class predictions

    # Step 5: Analyze predictions
    has_fallen = 1 in predictions  # Check if any frame predicts 'fallen' (class 1)
    frame_results = {f"Frame {i * frame_skip + 1}": "Fallen" if pred == 1 else "Not Fallen"
                     for i, pred in enumerate(predictions)}

    # Return results
    return {
        "Frame Results": frame_results,
        "Overall Result": "Fall Detected" if has_fallen else "No Fall Detected"
    }
```
:::

::: {.cell .markdown id="gSe6ODUOsowA"}
##Execution
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":73}" id="hFXKr4ZK4X4T" outputId="6fedbc0a-770a-48bf-b2fe-c43025b2c42f"}
``` python
from google.colab import files

# Upload video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
```

::: {.output .display_data}
```{=html}

     <input type="file" id="files-1f18f11c-b461-4d17-92f7-f6060b6d81dd" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-1f18f11c-b461-4d17-92f7-f6060b6d81dd">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 
```
:::

::: {.output .stream .stdout}
    Saving person_walking.mp4 to person_walking.mp4
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="90IBecg_iF7-" outputId="c9d8a8a0-c59c-42f6-896e-7ea6390ef834"}
``` python
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
```

::: {.output .stream .stdout}
    Using GPU: Tesla T4
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="C2hLEXvVPQ2q" outputId="40b9ccaa-8275-4a04-cd43-dbc9aee12165"}
``` python
import cv2
import mediapipe as mp
import numpy as np

# Path to your uploaded video
video_path = '/content/person_falling_new.mp4'  # Replace with your video path

# Open video using OpenCV
cap = cv2.VideoCapture(video_path)

# List to store frames
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Optionally preprocess (resize, etc.)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 for faster processing
    frames.append(frame_resized)

cap.release()

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# List to store keypoints for each frame
keypoints_list = []

for frame in frames:
    # Convert frame to RGB (MediaPipe works with RGB frames)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to extract pose landmarks
    results = pose.process(frame_rgb)

    # Extract keypoints if detected
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        keypoints_list.append(keypoints)
    else:
        keypoints_list.append(None)  # If no pose is detected, append None

pose.close()

# Print the number of frames and some example keypoints for debugging
print(f"Total frames: {len(frames)}")
print(f"First frame keypoints (if available): {keypoints_list[0]}")
```

::: {.output .stream .stdout}
    Total frames: 199
    First frame keypoints (if available): [[0.3686886131763458, 0.019866913557052612, -0.2704041600227356], [0.3765782117843628, 0.001132369041442871, -0.2532722055912018], [0.3814921975135803, 0.001101166009902954, -0.25336506962776184], [0.386398583650589, 0.0012714564800262451, -0.2534179091453552], [0.3635946214199066, -5.647540092468262e-05, -0.24685418605804443], [0.35973626375198364, -0.000982135534286499, -0.2468380481004715], [0.3561495542526245, -0.001795053482055664, -0.24692454934120178], [0.39508256316185, 0.008215129375457764, -0.11848311126232147], [0.35692110657691956, 0.005233645439147949, -0.08810247480869293], [0.377853661775589, 0.034045781940221786, -0.21703632175922394], [0.3624286651611328, 0.032154083251953125, -0.20829716324806213], [0.44082823395729065, 0.11007609963417053, -0.045047711580991745], [0.32432693243026733, 0.12767410278320312, 0.02514456771314144], [0.46256619691848755, 0.2580544650554657, -0.012743744067847729], [0.3108488619327545, 0.2689981460571289, 0.09872337430715561], [0.4494265019893646, 0.37630170583724976, -0.16853441298007965], [0.2980857193470001, 0.4020870327949524, -0.0303350742906332], [0.44897836446762085, 0.4174494743347168, -0.21934838593006134], [0.2923118770122528, 0.4451368749141693, -0.07158957421779633], [0.4352758824825287, 0.41113823652267456, -0.26141777634620667], [0.29777252674102783, 0.4502367675304413, -0.12590435147285461], [0.4322224259376526, 0.4005734622478485, -0.18774063885211945], [0.30212271213531494, 0.4380636215209961, -0.055740367621183395], [0.4128381609916687, 0.40266624093055725, -0.019901666790246964], [0.3429158627986908, 0.3987671732902527, 0.020087342709302902], [0.3972861170768738, 0.6304139494895935, -0.08849058300256729], [0.33577820658683777, 0.6290516257286072, -0.050471507012844086], [0.3906082510948181, 0.8325799703598022, 0.13235993683338165], [0.3371073603630066, 0.837647557258606, 0.14519518613815308], [0.38703858852386475, 0.8422458171844482, 0.14348363876342773], [0.33949193358421326, 0.8537224531173706, 0.15369153022766113], [0.3869437575340271, 0.9205210208892822, -0.04904864355921745], [0.3281879723072052, 0.9166303873062134, -0.03785967081785202]]
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="r29WsGgf628_" outputId="9fb4159e-ce83-444d-f06d-fcf85774b923"}
``` python
import cv2
import mediapipe as mp
import numpy as np

# Path to your uploaded video
video_path = '/content/person_walking.mp4'  # Replace with your video path

# Open video using OpenCV
cap = cv2.VideoCapture(video_path)

# List to store frames
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Optionally preprocess (resize, etc.)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 for faster processing
    frames.append(frame_resized)

cap.release()

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# List to store keypoints for each frame
keypoints_list = []

for frame in frames:
    # Convert frame to RGB (MediaPipe works with RGB frames)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to extract pose landmarks
    results = pose.process(frame_rgb)

    # Extract keypoints if detected
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        keypoints_list.append(keypoints)
    else:
        keypoints_list.append(None)  # If no pose is detected, append None

pose.close()

# Print the number of frames and some example keypoints for debugging
print(f"Total frames: {len(frames)}")
print(f"First frame keypoints (if available): {keypoints_list[3]}")
```

::: {.output .stream .stdout}
    Total frames: 178
    First frame keypoints (if available): [[0.48274102807044983, 0.524723470211029, 0.06934764981269836], [0.48136845231056213, 0.5177784562110901, 0.05716276913881302], [0.48001083731651306, 0.517755389213562, 0.05714453384280205], [0.47871294617652893, 0.5178438425064087, 0.057123295962810516], [0.4847719073295593, 0.5178069472312927, 0.054578136652708054], [0.48585644364356995, 0.5179035663604736, 0.05459362268447876], [0.4870249629020691, 0.5181229114532471, 0.05455111712217331], [0.47675877809524536, 0.5212246179580688, 0.01810361072421074], [0.48929929733276367, 0.5222467184066772, 0.005639493931084871], [0.4805240035057068, 0.5304532647132874, 0.058937229216098785], [0.484630286693573, 0.531627893447876, 0.05520838499069214], [0.4577752649784088, 0.5603043437004089, -0.0005099961417727172], [0.4972974359989166, 0.5665785074234009, -0.013322453014552593], [0.45488423109054565, 0.6230266094207764, -0.008424973115324974], [0.5024628043174744, 0.6276262402534485, -0.030028587207198143], [0.4576384127140045, 0.6678259372711182, -0.011452739126980305], [0.5027928948402405, 0.6723214983940125, -0.044642288237810135], [0.45842140913009644, 0.6814358830451965, -0.015885597094893456], [0.502453625202179, 0.686213493347168, -0.050857119262218475], [0.4636403024196625, 0.6801166534423828, -0.020046215504407883], [0.4997529983520508, 0.682561993598938, -0.05594164878129959], [0.464484840631485, 0.6751745343208313, -0.013372073881328106], [0.4986080229282379, 0.6772783994674683, -0.047479432076215744], [0.4666527211666107, 0.6754019260406494, 0.008676846511662006], [0.48813194036483765, 0.6788484454154968, -0.00870196707546711], [0.46963509917259216, 0.7484113574028015, 0.06568063795566559], [0.4828987121582031, 0.7591772079467773, 0.012363152578473091], [0.467691570520401, 0.8192822337150574, 0.13764142990112305], [0.47816362977027893, 0.8405625224113464, 0.04527779668569565], [0.4671541154384613, 0.8321870565414429, 0.14369350671768188], [0.47694215178489685, 0.8505257368087769, 0.04729278013110161], [0.47004371881484985, 0.8197529911994934, 0.11711569130420685], [0.4793100357055664, 0.8509629964828491, 0.008653700351715088]]
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="uPbDeIb9SAeW" outputId="1d93f056-b094-40e7-ca16-56df4bc90e64"}
``` python
# Initialize your model
model = STGCN(num_keypoints=len(keypoints_list), num_classes=2)  # Assuming 33 keypoints and 2 classes  # Use your actual model class name here
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved state_dict
model_path = '/content/drive/MyDrive/STGCN_Dataset/stgcn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)


# Set the model to evaluation mode
model.eval()
```

::: {.output .execute_result execution_count="56"}
    STGCN(
      (stgcn1): STGCNLayer(
        (spatial_conv): Conv2d(1, 64, kernel_size=(1, 178), stride=(1, 1), padding=(0, 89))
        (temporal_conv): Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (stgcn2): STGCNLayer(
        (spatial_conv): Conv2d(64, 64, kernel_size=(1, 178), stride=(1, 1), padding=(0, 89))
        (temporal_conv): Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (fc): Linear(in_features=64, out_features=2, bias=True)
    )
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="yj6SEYAllV_g" outputId="e1f75b06-0d6d-4873-8552-3cb392b7a4de"}
``` python
import torch
import cv2
import numpy as np
import mediapipe as mp

# Define pose extraction function using MediaPipe
def pose_extractor(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Extract keypoints
    if results.pose_landmarks:
        keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                              for landmark in results.pose_landmarks.landmark])
        return keypoints
    else:
        return None  # No keypoints detected

# Call the prediction function with video path and model
video_path = '/content/person_falling_new.mp4'  # Replace with your video path
results = predict_fall_in_video(video_path, model, device, pose_extractor)

# Print the frame results and overall result
print("Per-frame results:", results['Frame Results'])
print("Overall result:", results['Overall Result'])
```

::: {.output .stream .stdout}
    Per-frame results: {'Frame 1': 'Fallen'}
    Overall result: Fall Detected
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="tbYDrUac419t" outputId="52444148-0001-4387-8903-92289ff17373"}
``` python
import torch
import cv2
import numpy as np
import mediapipe as mp

# Define pose extraction function using MediaPipe
def pose_extractor2(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Extract keypoints
    if results.pose_landmarks:
        keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                              for landmark in results.pose_landmarks.landmark])
        return keypoints
    else:
        return None  # No keypoints detected

# Call the prediction function with video path and model
video_path2 = '/content/person_walking.mp4'  # Replace with your video path
results = predict_fall_in_video(video_path2, model, device, pose_extractor2)

# Print the frame results and overall result
print("Per-frame results:", results['Frame Results'])
print("Overall result:", results['Overall Result'])
```

::: {.output .stream .stdout}
    Per-frame results: {'Frame 1': 'Fallen'}
    Overall result: Fall Detected
:::
:::

::: {.cell .markdown id="aDrFX3fRiigL"}
#Twilio Integration with G-Drive
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="zmi1W3sVP3AS" outputId="7c366a6b-c689-4856-819f-20a9933c6cdc"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```

::: {.output .stream .stdout}
    Mounted at /content/drive
:::
:::

::: {.cell .code id="1KxUBMraQJ_N"}
``` python
import cv2
import os

def save_to_gdrive_and_get_link(frame, file_name, gdrive_folder='MyDrive/Fall_Detection_Alerts'):
    """
    Saves the frame to Google Drive and retrieves a shareable link.

    Parameters:
        frame (numpy.ndarray): Image frame to save.
        file_name (str): Name for the saved file.
        gdrive_folder (str): Folder in Google Drive where the file will be saved.

    Returns:
        str: Google Drive shareable link for the saved file.
    """
    # Ensure Google Drive folder exists
    folder_path = f"/content/drive/{gdrive_folder}"
    os.makedirs(folder_path, exist_ok=True)

    # Save frame to Google Drive
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, frame)
    print(f"Frame saved to: {file_path}")

    # Return the relative path in Google Drive for easier access
    relative_file_path = os.path.join(gdrive_folder, file_name)
    print(f"Manual location: {relative_file_path}")

    return relative_file_path  # Return relative path within MyDrive
```
:::

::: {.cell .code id="r5lLOjKvQZuS"}
``` python
def send_fall_notification_via_gdrive_simple(prediction_results, video_path, account_sid, auth_token, from_number, to_number):
    from twilio.rest import Client

    # Extract the overall result
    overall_result = prediction_results["Overall Result"]

    # Initialize Twilio client
    client = Client(account_sid, auth_token)

    if overall_result == "Fall Detected":
        # Extract a key frame from the video
        cap = cv2.VideoCapture(video_path)
        fallen_frame_index = None
        for i, (frame_key, frame_result) in enumerate(prediction_results["Frame Results"].items()):
            if frame_result == "Fallen":
                fallen_frame_index = i
                break

        # Extract and save the fallen frame
        if fallen_frame_index is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallen_frame_index)
            ret, frame = cap.read()
            if ret:
                # Save to Google Drive
                fallen_frame_path = save_to_gdrive_and_get_link(frame, "fallen_frame.jpg")
                # Send MMS with a manual location
                message = client.messages.create(
                    body=f"Fall detected! Immediate assistance may be needed. Check Google Drive for the alert frame: {fallen_frame_path}",
                    from_=from_number,
                    to=to_number
                )
                print(f"Fall alert sent via SMS: SID {message.sid}")
            else:
                print("Failed to extract fallen frame.")
        else:
            print("No specific fallen frame detected in video.")
        cap.release()
    else:
        # Send a simple SMS indicating no fall was detected
        message = client.messages.create(
            body="Patient is okay. No fall detected.",
            from_=from_number,
            to=to_number
        )
        print(f"No fall detected notification sent: SID {message.sid}")
```
:::

::: {.cell .code id="u52MQ6aVT5Xz"}
``` python
# Dummy prediction results for testing
fall_detected_results = {
    "Frame Results": {
        "Frame 1": "Not Fallen",
        "Frame 2": "Fallen",
        "Frame 3": "Fallen",
    },
    "Overall Result": "Fall Detected"
}

no_fall_detected_results = {
    "Frame Results": {
        "Frame 1": "Not Fallen",
        "Frame 2": "Not Fallen",
        "Frame 3": "Not Fallen",
    },
    "Overall Result": "No Fall Detected"
}

# Dummy video path
video_path = "person_falling_new.mp4"  # Replace this with the path to your test video file
```
:::

::: {.cell .markdown id="wMr0UKRk7JBw"}
##Twilio Credentials
:::

::: {.cell .code id="EVC_ukS07LpS"}
``` python
# Twilio account details (replace with your actual credentials)
account_sid = "AC4711d8b017004bc588e6141e749cdf60"
auth_token = "ec36ccc0fad09bd6e411d2c7f8337006"
from_number = "whatsapp:+14155238886"
to_number = "whatsapp:+918593811202"
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iDosNcG1T8Uc" outputId="0461a8d7-70da-42e8-d017-3e7e673f24ec"}
``` python
# Fall Detected Case
print("Testing Fall Detected Case:")
send_fall_notification_via_gdrive_simple(
    fall_detected_results,
    video_path,
    account_sid,
    auth_token,
    from_number,
    to_number
)
```

::: {.output .stream .stdout}
    Testing Fall Detected Case:
    Frame saved to: /content/drive/MyDrive/Fall_Detection_Alerts/fallen_frame.jpg
    Manual location: MyDrive/Fall_Detection_Alerts/fallen_frame.jpg
    Fall alert sent via SMS: SID SMf053af337749a6963b5161d753deda26
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="vqkWc22qT-h6" outputId="5a72169e-6a9e-4083-99b1-1fd584d5c0e2"}
``` python
# No Fall Detected Case
print("\nTesting No Fall Detected Case:")
send_fall_notification_via_gdrive_simple(
    no_fall_detected_results,
    video_path,
    account_sid,
    auth_token,
    from_number,
    to_number
)
```

::: {.output .stream .stdout}

    Testing No Fall Detected Case:
    No fall detected notification sent: SID SM86a22a9d076522ae109d380a82aae1c5
:::
:::
