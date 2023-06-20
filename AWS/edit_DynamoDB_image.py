import boto3
import json

def lambda_handler(event, context):
    
    dynamodb = boto3.client('dynamodb')

    
    table_name = 'kaixu'

    
    id_list = []
    img_url_list = []
    tags_list = []

    
    response = dynamodb.scan(TableName=table_name)
    dynamodb1 = boto3.resource('dynamodb')
    table = dynamodb1.Table('kaixu')

   
    for item in response['Items']:
        
        id_list.append(item['id']['N'])
        img_url_list.append(item['img_url']['S'])
        tags_list.append(item['tags']['S'])
    
    target_url = event.get('url')
    taget_type = event.get('type')
    taget_tags = event.get('tags')
    print(target_url)
    url_index = []
    taget_list = []
    
    for index, url in enumerate(img_url_list):
        if url == target_url:
            url_index.append(index)
    
    label_list = [{str(label['count']): label['tag']} for label in taget_tags]
    
    
    json_data = json.dumps(label_list)
    json_data = json.loads(json_data)
    
    if url_index != []:
        for i in url_index:
            newtag = []
            id = id_list[i]
            url = img_url_list[i]
            tags = tags_list[i]
            tags = json.loads(tags)
            
            newtag.append(calculate(tags, taget_type, json_data))
            result = []
            
            for item in newtag:
                for key, value in item.items():
                    result.append({str(value): key})
            
            new_item = {
                'id': int(id),
                'img_url': url,
                'tags': json.dumps(result)
            }
            
            table.delete_item(Key={'id': int(id), 'img_url': url})
            table.put_item(Item=new_item)
    
    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'after_edit': "successful"
    }
    
    return response

def calculate(tags, target_type, json_data):
    result_dict = {}
    result_dict1 = {}
    merged_dict = {}  
    
    for item in json_data:
        for key, value in item.items():
            result_dict[value] = int(key)
    
    if tags != []:
        for item1 in tags:
            for key1, value1 in item1.items():
                result_dict1[value1] = int(key1)
                
                if target_type == 1:
                    merged_dict = result_dict1.copy()
                    
                    for key, value in result_dict.items():
                        if key in merged_dict:
                            merged_dict[key] += value
                        else:
                            merged_dict[key] = value
                else:
                    merged_dict = result_dict1.copy()
                    
                    for key, value in result_dict.items():
                        if key in merged_dict:
                            merged_dict[key] -= value
                        else:
                            merged_dict[key] = -value
                    
                    
                    merged_dict = {key: value for key, value in merged_dict.items() if value > 0}
    else:
        if target_type == 1:
            merged_dict = result_dict1.copy()
            
            for key, value in result_dict.items():
                if key in merged_dict:
                    merged_dict[key] += value
                else:
                    merged_dict[key] = value
        else:
            merged_dict = {}
            
    return merged_dict
