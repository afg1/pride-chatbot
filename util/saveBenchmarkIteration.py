import requests
import json

# URL to fetch JSON data from
url = "http://codon-gpu-016.ebi.ac.uk:6008/getBenchmark?page_num=0&items_per_page=200&iteration=4"

try:
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON data from the response
        data = response.json()

        # Specify the filename to save the JSON data
        filename = "../resources/4thIteration.json"

        # Save the JSON data to a file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"JSON data has been successfully saved to {filename}")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON data: {e}")