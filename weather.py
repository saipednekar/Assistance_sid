import requests
def w():

    url = "https://aerisweather1.p.rapidapi.com/observations/Goa,in"

    headers = {
        "X-RapidAPI-Key": "cc09c2dff5mshe9da178defb9fe2p177ad5jsnd07927977aed",
        "X-RapidAPI-Host": "aerisweather1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers)
    res=response.json()
    res=str(res["response"]["ob"]["weather"])
    return res 