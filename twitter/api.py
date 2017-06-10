import base64
from functools import reduce
from os import getenv

import grequests
import requests


KEY = getenv('TSABL_CONSUMER_KEY')
SECRET = getenv('TSABL_CONSUMER_SECRET')


class TwitterApi:
    def __init__(self):
        self.token = self.get_token()

    @staticmethod
    def base64_encode(string):
        return base64.b64encode(bytes(string, 'utf-8')).decode('utf-8')

    def get_token(self):
        r = requests.post('https://api.twitter.com/oauth2/token',
                          headers={
                              'Authorization': 'Basic ' + self.base64_encode(KEY + ':' + SECRET),
                              'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
                          },
                          data={
                              'grant_type': 'client_credentials'
                          })
        json_response = r.json()
        r.raise_for_status()
        if 'errors' in json_response:
            return json_response
        if r.status_code == requests.codes.ok:
            return json_response['access_token']

    # Asynchronously fetches tweet objects for corresponding ids
    def bulk_get_statuses(self, _ids):
        ids = _ids[:]

        # Split ids into bulks of max length 100, because this is max for the Twitter REST api
        bulks = []
        while len(ids) > 0:
            bulks.append(ids[:100])
            ids = ids[100:]

        url = 'https://api.twitter.com/1.1/statuses/lookup.json'

        # Function that creates a request for a given bulk
        def fetch_bulk(bulk):
            return grequests.post(url,
                                  headers={
                                      'Authorization': 'Bearer ' + self.token
                                  },
                                  params={
                                      'id': ','.join(bulk),
                                      'trim_user': True
                                  })

        # Execute requests in parallel
        unsent_reqs = (fetch_bulk(bulk) for bulk in bulks)
        executed_reqs = map(lambda r: r.json(), grequests.map(unsent_reqs))
        flattened_list = reduce(lambda a, b: a + b, executed_reqs)
        return list(flattened_list)
