import boto3
import json
from boto3.dynamodb.conditions import Key, Attr


def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('kaixu')
    s3 = boto3.resource('s3')
    bucket_name = 'image-uploadass2'
    img_name = event.get('img_name')
    img_url = 'https://image-uploadass2.s3.amazonaws.com/'+ img_name
    delete_from_s3(img_url, bucket_name)
    itemid,imgurl = delete_from_dynamodb(img_url, table)
    response = {
                'statusCode': 200,
                'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'delect_id': itemid,
                'delect_url':imgurl
            }
    return response


def delete_from_s3(img_url, bucket_name):
    s3 = boto3.resource('s3')
    object_key = img_url.split('/')[-1]
    s3.Object(bucket_name, object_key).delete()

def delete_from_dynamodb(img_url, table):
    response = table.scan(FilterExpression=Attr('img_url').eq(img_url))
    if response['Items']:
        for item in response['Items']:
            item_id = item.get('id')
            table.delete_item(Key={'id': int(item_id), 'img_url': img_url})
            return item_id,img_url

