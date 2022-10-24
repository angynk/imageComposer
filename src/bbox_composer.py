import torch
from xml.etree import ElementTree as et


def get_bbox_usual_resize(annot_file_path, target_width, target_height, classes, width, height):
    
    boxes = []
    labels = []
    tree = et.parse(annot_file_path)
    root = tree.getroot()

    # TEMPORAL -Just for bad annottations-
    object = False

    # box coordinates for xml files are extracted and corrected for image size given
    for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        labels.append(classes.index(member.find('name').text))
            
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)
            
        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        xmin_final = (xmin/target_width)*width
        xmax_final = (xmax/target_width)*width
        ymin_final = (ymin/target_height)*height
        yamx_final = (ymax/target_height)*height
            
        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
    

        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    return boxes