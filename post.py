# from fdfs_client.client import Fdfs_client
#
# client = Fdfs_client('client.conf')
# res = client.upload_by_filename('test.py')
# print(res)
import requests
r = requests.post("http://0.0.0.0:16001/bp_model_build", data={'model_id':"1dfhsdhfksdjfh",'train_data1':"sldkfjldskjf"})
print(r.text)