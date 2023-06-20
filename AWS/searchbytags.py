import boto3
import json
def lambda_handler(event, context):

    dynamodb = boto3.client('dynamodb')
    target_tags = event.get("tags", [])
    print(target_tags)

        

    table_name = 'kaixu'
    tag_value = 'person'
    count_value = 1

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
    response = {
                'statusCode': 200,
                'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'links': json.dumps({'links': links})
            }
    return response
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


   