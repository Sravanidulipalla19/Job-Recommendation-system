import boto3
import json
import sys
s3 = boto3.client('s3')

BUCKET_NAME = 'job-recommendation-system-cleanse-useast-1-69522247-dev'

# get jobs information from single json file
def get_jobs_from_key(bucket_name, key):

    response = s3.get_object(
        Bucket=bucket_name,
        Key=key,
    )
    jobs_list = json.loads(response['Body'].read().decode('utf-8'))
    return jobs_list

# accumulate jobs information between certain periods
def get_jobs_from_prefix(bucket_name, prefix):
    return_jobs_list = []
    response = s3.list_objects(
        Bucket='job-recommendation-system-cleanse-useast-1-69522247-dev',
        Prefix='2022/07/',
    )

    for json_file in response['Contents']:
        key = json_file['Key']
        jobs_list = get_jobs_from_key(BUCKET_NAME, key)
        if len(jobs_list) != 0:
            return_jobs_list.extend(jobs_list)

    return return_jobs_list


for job in get_jobs_from_prefix(BUCKET_NAME, '2022/07/'):
    print(job)
