from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("vit-0mr4g").project("auto-rickshaw-oobkc")
dataset = project.version(2).download("yolov8")
print(dataset.location)