import os
import json
import datetime
import uuid
import yaml
import cv2
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
HISTORY_FILE = 'history.json'
DATA_YAML_PATH = 'data.yaml'  # Путь к вашему YAML с описанием классов

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Загружаем имена классов из data.yaml
with open(DATA_YAML_PATH, 'r') as f:
    data_config = yaml.safe_load(f)
class_names = data_config.get('names', [])

# Загружаем модель с вашими весами
model = YOLO('best.pt')

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # Проверка на 0 или NaN
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 для браузера
    output_filename = 'processed_' + os.path.basename(filepath)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter для файла {output_path}")

    unique_ids = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.size == 0:
            continue

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            for i in range(len(track_ids)):
                # Считаем все классы как дефекты
                unique_ids.add(track_ids[i])

        # Записываем кадр без изменения цветового пространства
        out.write(annotated_frame)

    cap.release()
    out.release()

    if os.path.getsize(output_path) == 0:
        raise RuntimeError("Видео не содержит записанных кадров")

    return output_filename, len(unique_ids)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = str(uuid.uuid4()) + '_' + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = os.path.splitext(filename)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']

    try:
        if is_video:
            output_filename, defect_count = process_video(filepath)
        else:
            results = model(filepath)
            unique_ids = set()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Считаем все объекты как дефекты
                for tid in track_ids:
                    unique_ids.add(tid)
            defect_count = len(unique_ids)

            annotated_frame = results[0].plot()
            output_filename = 'annotated_' + filename
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(annotated_filepath, annotated_frame)
    except Exception as e:
        history = load_history()
        history.append({
            'date': datetime.datetime.now().isoformat(),
            'file': filename,
            'error': str(e),
            'is_video': is_video
        })
        save_history(history)
        return f"Ошибка обработки: {str(e)}"

    history = load_history()
    history.append({
        'date': datetime.datetime.now().isoformat(),
        'file': output_filename,
        'count_defect': defect_count,
        'is_video': is_video
    })
    save_history(history)

    return render_template('result.html', filename=output_filename, count=defect_count, is_video=is_video)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.mp4', '.avi', '.mov']:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')
    elif ext in ['.jpg', '.jpeg', '.png']:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='image/jpeg')
    else:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/report')
def generate_report():
    history = load_history()
    if not history:
        return "Нет данных для отчёта"

    defect_json_path = os.path.join(app.config['REPORT_FOLDER'], 'defect_information.json')
    with open(defect_json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    report_path = os.path.join(app.config['REPORT_FOLDER'], 'defect_information.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Defect report", styles['Title']))

    data = [['Date', 'File', 'Type', 'Defect Count']]
    for entry in history:
        date = entry.get('date', '-')
        file = entry.get('file', '-')
        file_type = "Video" if entry.get('is_video', False) else "Picture"
        count = str(entry.get('count_defect', 0))
        data.append([date, file, file_type, count])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#cce5ff')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f7fbff')),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(table)

    doc.build(elements)
    return send_file(report_path, as_attachment=True, mimetype='application/pdf')

@app.route('/clear')
def clear_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

    for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORT_FOLDER']]:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                print(f"Ошибка при удалении файла {path}: {e}")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    