import cv2
import numpy as np
import time
import os
import threading
import tempfile
from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from ollama import chat

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

ocorrencias = 0
analise_iniciada = False
ai_response_text = ""
text_lock = threading.Lock()
last_analysis_time = 0.0
ANALYSIS_COOLDOWN = 10.0
# NOTA: Baixe o arquivo 'DejaVuSans.ttf' e coloque-o no mesmo diretório deste script.
FONT_PATH = "DejaVuSans.ttf" 

def analisar_imagem_gemini(imagem_input, prompt="Descreva esta imagem em uma frase curta, foque em dar o contexto da situação para um sistema de segurança."):
    print("Analisando imagem com Gemini...")
    try:
        if isinstance(imagem_input, str):
            imagem = Image.open(imagem_input)
        elif isinstance(imagem_input, np.ndarray):
            rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
            imagem = Image.fromarray(rgb)
        else:
            raise ValueError("Tipo de entrada não suportado.")

        modelo = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        resposta = modelo.generate_content([prompt, imagem])
        return resposta.text
    except Exception as e:
        return f"Erro Gemini: {e}"

def analisar_imagem_ollama(imagem_input, prompt="Descreva esta imagem em uma frase curta, foque em dar o contexto da situação para um sistema de segurança."):
    temp_path = None
    print("Analisando imagem com Ollama...")
    try:
        if isinstance(imagem_input, str):
            path_para_ollama = imagem_input
        elif isinstance(imagem_input, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                img_pil.save(temp_path, format="PNG")
                path_para_ollama = temp_path
        else:
            raise ValueError("Tipo de entrada não suportado")

        response = chat(
            model="qwen2.5vl:3b",
            messages=[{"role": "user", "content": prompt, "images": [path_para_ollama]}],
        )
        resposta = response['message']['content']
        
        return resposta
    except Exception as e:
        return f"Erro Ollama: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def worker_analisar(frame_para_analisar):
    global ai_response_text
    #texto_descricao = analisar_imagem_ollama(frame_para_analisar)
    
    texto_descricao = analisar_imagem_gemini((frame_para_analisar))
    
    with text_lock:
        #ai_response_text = f"[Qwen]: {texto_descricao}"
        ai_response_text = f"[Gemini] {texto_descricao}"
       
        print(ai_response_text)

def draw_wrapped_text_utf8(frame, text, org, font_path, font_size, color, max_width):
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except IOError:
        cv2.rectangle(frame, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        error_msg = "ERRO: Fonte .ttf nao encontrada. Caracteres especiais nao serao exibidos."
        cv2.putText(frame, error_msg, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)

    line_height = font.getbbox("Tg")[3] + 5
    total_text_height = len(lines) * line_height
    rect_y1 = org[1] - total_text_height
    
    draw.rectangle(((0, rect_y1 - 10), (frame.shape[1], org[1] + 10)), fill=(0, 0, 0))

    y = rect_y1
    for line in lines:
        draw.text((org[0], y), line, font=font, fill=color)
        y += line_height

    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

model = YOLO('yolo11n.pt')
classes_desejadas = [0, 1, 2, 3, 4]
CONF_THR = 0.5
DIST_THR = 20
FRAMES_LIMIT = 5
ALERT_DURATION = 3.0

prev_centroid = None
same_region_count = 0
alert_end_time = 0
alert_class_name = ''

input_path = 'entregador.mp4'
output_path = 'output_entregador_GEMINI_UTF8.mp4'

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Não foi possível abrir o vídeo de entrada")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise IOError("Não foi possível criar o vídeo de saída")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if analise_iniciada and (time.time() - last_analysis_time > ANALYSIS_COOLDOWN):
        analise_iniciada = False
        ocorrencias = 0

    results = model.predict(source=frame, classes=classes_desejadas, conf=CONF_THR, stream=False, verbose=False)

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        cls_id = int(results[0].boxes.cls[0].cpu().numpy())
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = box
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        if prev_centroid is not None:
            dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if dist < DIST_THR:
                same_region_count += 1
            else:
                same_region_count = 0
        prev_centroid = centroid

        if same_region_count >= FRAMES_LIMIT and time.time() > alert_end_time:
            alert_end_time = time.time() + ALERT_DURATION
            alert_class_name = class_name
            ocorrencias += 1
            if ocorrencias == 2 and not analise_iniciada:
                analise_iniciada = True
                last_analysis_time = time.time()
                frame_copy = frame.copy()
                thread = threading.Thread(target=worker_analisar, args=(frame_copy,))
                thread.daemon = True
                thread.start()

    if results:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if time.time() < alert_end_time:
        cv2.putText(frame, f'State: {alert_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    with text_lock:
        local_text = ai_response_text
    if local_text:
        frame = draw_wrapped_text_utf8(frame, local_text, org=(10, height - 20), font_path=FONT_PATH, font_size=20, color=(255, 255, 255), max_width=width - 20)

    writer.write(frame)
    cv2.imshow('Deteccao YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()