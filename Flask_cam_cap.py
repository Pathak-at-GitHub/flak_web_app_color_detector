from flask import Flask , render_template, Response
import cv2
from util_for_color import get_limits
from PIL import Image

app = Flask(__name__, template_folder='../FDIP_Lab_MiniProject')

camera = cv2.VideoCapture(0)
yellow = [255, 255,0]
def generate_frame():
    while True:
        succecc, frame = camera.read()
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_limit, upper_limit = get_limits(yellow)

        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, x2, y1, y2 = bbox
            frame = cv2.rectangle(frame, (x1, x2), (y1, y2), (0, 255, 0), 5)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not succecc:
            break;
        else:
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
