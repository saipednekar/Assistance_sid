import requests
def w():

    url = "https://aerisweather1.p.rapidapi.com/observations/"

    headers = {
        "X-RapidAPI-Key": "",
        "X-RapidAPI-Host": ""
    }

    response = requests.request("GET", url, headers=headers)
    res=response.json()
    res=str(res["response"]["ob"]["weather"])
    return res 
