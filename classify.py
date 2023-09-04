import onnxruntime as ort

model = ort.InferenceSession("yolov8m.onnx")

inputs = model.get_inputs();
input = inputs[0]
print("Input Name:",input.name)
print("Input Type:",input.type)
print("Input Shape:",input.shape)

outputs = model.get_outputs()
output = outputs[0]
print("Output Name:",output.name)
print("Output Type:",output.type)
print("Output Shape:",output.shape)

from PIL import Image

img = Image.open("cat-dog.jpg")
img_width, img_height = img.size;
img = img.resize((640,640))
img = img.convert("RGB")

import numpy as np

input = np.array(img)
input = input.transpose(2,0,1)
input = input.reshape(1,3,640,640)
input = input/255.0
input = input.astype(np.float32)

outputs = model.run(["output0"], {"images":input})
output = outputs[0]
output = output[0]
output = output.transpose()

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def parse_row(row):
    xc,yc,w,h = row[:4]
    x1 = (xc-w/2)/640*img_width
    y1 = (yc-h/2)/640*img_height
    x2 = (xc+w/2)/640*img_width
    y2 = (yc+h/2)/640*img_height
    prob = row[4:].max()
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1,y1,x2,y2,label,prob]

boxes = [row for row in [parse_row(row) for row in output] if row[5]>0.5]

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

boxes.sort(key=lambda x: x[5], reverse=True)
result = []
while len(boxes)>0:
    result.append(boxes[0])
    boxes = [box for box in boxes if iou(box,boxes[0])<0.7]

from PIL import ImageDraw
img = Image.open("cat-dog.jpg")
draw = ImageDraw.Draw(img)

for box in result:
    x1,y1,x2,y2,class_id,prob = box
    draw.rectangle((x1,y1,x2,y2),None,"#00ff00")

img.show()
