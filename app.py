from flask import Flask, render_template, request
import requests
from pyproj import Transformer
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)


def geocode_address_nominatim(address):
    """ Get latitude and longitude using Nominatim API """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "YourAppName/1.0 (your-email@example.com)"  # Replace with your app's name and email
    }
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error in geocoding: {response.status_code}")

    data = response.json()
    if not data:
        raise ValueError("Address not found.")

    location = data[0]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832")
    bbox = [float(l) for l in location["boundingbox"]]
    return transformer.transform(bbox[:2], bbox[2:])


def expand_bounding_box(bbox, e_v=10):
    (north, south), (east, west) = bbox
    return [north - e_v, south + e_v], [east - e_v, west + e_v]


def getImage(latitude, longitude, year=2024):
    url = f"https://geodaten.metropoleruhr.de/dop/dop_{year}?language=ger&_signature=33%3A6N8JFskQtWDZ-SNr0ZtLFbSnY1M&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&FORMAT=image%2Fjpeg&TRANSPARENT=FALSE&LAYERS=dop_{year}&STYLES=&_OLSALT=0.09943999623862132&WIDTH=512&HEIGHT=512&CRS=EPSG%3A25832&BBOX={latitude[0]},{longitude[0]},{latitude[1]},{longitude[1]}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to fetch image, status code: {response.status_code}")
        return None


def detect_new_house(img1, img2, threshold: float = 0.3):
    image1 = np.asarray(img1).copy()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.GaussianBlur(image1, (5, 5), 0)

    image2 = np.asarray(img2).copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.GaussianBlur(image2, (5, 5), 0)

    center_x = image1.shape[0] // 2
    center_y = image1.shape[1] // 2
    template_ratio = 0.9
    template_size = int(image1.shape[0] * template_ratio) // 2
    template = image1[center_x - template_size:center_x + template_size,
               center_y - template_size:center_y + template_size]  # Example: Crop a region containing a house in image1
    result = cv2.matchTemplate(image2, template, cv2.TM_CCOEFF_NORMED)

    # Step 3: Threshold the result to find the regions where the template matches

    locations = np.where((np.max(result) == result) & (np.max(result) > threshold))

    # Step 4: Draw rectangles around the detected houses
    output = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    for pt in zip(*locations[::-1]):  # Locations are given in (row, col) order, so we reverse it
        cv2.rectangle(output, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

    return output, np.max(result)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        address = request.form['address']
        year1 = request.form['year1']
        year2 = request.form['year2']
        try:
            bbox = geocode_address_nominatim(address)
            bbox = lat, lon = expand_bounding_box(bbox)
            years = [int(year1), int(year2)]
            imgs = [getImage(lat, lon, year=y) for y in years]

            ref_img = imgs[-1]
            threshold = 0.3
            diff_image, max_val = detect_new_house(imgs[-1], imgs[-2], threshold)
            new_house_detected = "Yes" if max_val < threshold else "No"
            houseMatch = max_val
            return render_template('index.html', imgs=imgs, years=years, new_house_detected=new_house_detected,
                                   diff_image=diff_image, houseMatch = houseMatch, zip=zip)

        except ValueError as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

from io import BytesIO
import base64

def image_to_base64(img):
    buffer = BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.frombuffer(mode="RGB", size=(img.shape[0], img.shape[1]), data=img)
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.template_filter('to_base64')
def to_base64_filter(img):
    return image_to_base64(img)

if __name__ == '__main__':
    app.run(debug=True)
