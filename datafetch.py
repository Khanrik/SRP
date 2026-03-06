import requests

url = "https://api.dataforsyningen.dk/dhm_wcs_DAF"
params = {
    "service": "WCS",
    "request": "GetCapabilities",
    "token": "7b8b57a36691b0b37ed1e4db995e1429"
}

response = requests.get(url, params=params)

print(response.text)
