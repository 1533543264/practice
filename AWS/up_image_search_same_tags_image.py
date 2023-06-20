import boto3
import base64
import json
from botocore.exceptions import NoCredentialsError
import numpy as np
import time
import cv2
import re
from boto3.dynamodb.conditions import Key, Attr


confthres = 0
nmsthres = 0
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('kaixu')
s3 = boto3.client('s3')
def lambda_handler(event, context):
    findsamepicture = event.get("findsamepicture")
    labels_list, label_count, imageUrl = process_image(event, context)
    filename_image ='findsamepicture '+ event['findsamepicture']['filename']
    same_tags = [{"tag": key, "count": value} for key, value in label_count.items()]
    same = {
        "tags": same_tags
    }
    links = searchfunction(same)
    links_filename =  [re.findall(r"https://image-uploadass2\.s3\.amazonaws\.com/(\d+\.jpg)", link)[0] for link in links]
    response = {
                'statusCode': 200,
                'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'same':same,
                'links': json.dumps({'links': links})
            }
    return response

def process_image(event):
    upload_data = event.get("upload")
    findsamepicture = event.get("findsamepicture")
    try:

        image = base64.b64decode(event['findsamepicture']['image'])

        coco_path = s3.get_object(
            Bucket='yolo3ass2',
            Key='coco.names'
        )
        cfg_path = s3.get_object(
            Bucket='yolo3ass2',
            Key='yolov3-tiny.cfg'
        )
        weights_path = s3.get_object(
            Bucket='yolo3ass2',
            Key='yolov3-tiny.weights'
        )

        labels = get_labels(coco_path)
        cfg_data = get_config(cfg_path)
        weights = get_weights(weights_path)
        nets_model = load_model(cfg_data, weights)
        tags = event.get('tags', [])
        links = []
        response = table.scan(FilterExpression=Attr('id').gte(0))
        
        if response['Items'] is not None:
            for item in response['Items']:
                item_tags = item.get('tags', [])
                
                if all(tag['tag'] in item_tags for tag in tags):
                    links.append(item.get('img_url'))

        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels_list, label_count, imageUrl = do_prediction(image, nets_model, labels, event['findsamepicture']['image'])
        return labels_list, label_count, imageUrl 
    except Exception as e:
        print("Exception {}".format(e))
        raise e

def dyDBupload(label_count, filename):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('kaixu')
    print(f"label_count11={label_count}")

    response = table.scan(ProjectionExpression='id')
    max_id = max([item['id'] for item in response['Items']]) if 'Items' in response else 0

    tags = [{count: str(label)} for label, count in label_count.items()]
    

    json_data = json.dumps(tags)

    
    new_item = {
        'id': max_id + 1,
        'img_url': "https://image-uploadass2.s3.amazonaws.com/" + filename,
        'tags':json_data
       
    }
    
    response = table.put_item(Item=new_item)
    print(f"response={response}")
    return response

def get_labels(labels_path):

    labels = labels_path['Body'].read().decode('utf-8').strip().split("\n")
    print(labels)
    return labels


def get_weights(weights_path):

    weights = weights_path['Body'].read()
    return weights


def get_config(config_path):

    config = config_path['Body'].read()
    return config


def load_model(configpath, weightspath):


    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, labels, imageUrl):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO {:.6f} s".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)

    if len(idxs) > 0:
        labels_list = []
        for i in idxs.flatten():
            labels_list.append(labels[classIDs[i]])
        label_count = generate_label_count(labels_list)
    return labels_list, label_count, imageUrl


def generate_label_count(label_list):

    label_count = {}
    for label in label_list:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        label_count[label] = label_count[label]
        print(f"label_count={label_count}")
    return label_count
def searchfunction(event):
    target_tags = event.get("tags", [])
    dynamodb = boto3.client('dynamodb')
    table_name = 'kaixu'
        

    ids = []
    img_urls = []
    tags = []

    response = dynamodb.scan(TableName=table_name)

    for item in response['Items']:

        ids.append(item['id']['N'])
        img_urls.append(item['img_url']['S'])
        tags.append(item['tags']['S'])
    data = tags
    result = []
    for item in data:
        if item:
            json_data = json.loads(item)
            if len(json_data) > 0:
                sub_result = []
                for obj in json_data:
                    tag = list(obj.values())[0]
                    count = int(list(obj.keys())[0])
                    sub_result.append({'tag': tag, 'count': count})
                result.append(sub_result)
            else:
                result.append([])
        else:
            result.append([])
    list_index = []
    for i in range(len(result)):
        if compare_data(target_tags, result[i]) == True :
            list_index.append(i)
    links = []
    for i in list_index:
        links.append(img_urls[i])
    links = list(set(links))
    
    return links
def compare_data(test_data, new_data):
    test_tags = set(item['tag'] for item in test_data)
    new_tags = set(item['tag'] for item in new_data)
    
    if not test_tags.issubset(new_tags):
        return False
    
    for test_item in test_data:
        test_tag = test_item['tag']
        test_count = test_item['count']
        
        for new_item in new_data:
            if new_item['tag'] == test_tag and new_item['count'] >= test_count:
                break
        else:
            return False
    
    return True