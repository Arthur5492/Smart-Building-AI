import cv2
import numpy as np
import time
from ultralytics import YOLO


from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image


import threading

from ollama import chat
import tempfile


load_dotenv()


api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)


ocorrencias = 0
contexto_via_gemini = None


def analisar_imagem_gemini(imagem_input, prompt="Descreva esta imagem em uma frase curta, foque em dar o contexto da situação para um sistema de segurança."):
    try:
       
        if isinstance(imagem_input, str):
            imagem = Image.open(imagem_input)
       
        elif isinstance(imagem_input, np.ndarray):
            rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
            imagem = Image.fromarray(rgb)
        else:
            raise ValueError("Tipo de entrada não suportado. Passe caminho (str) ou frame (np.ndarray).")

       
        modelo = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        resposta = modelo.generate_content([prompt, imagem])
        return resposta.text

    except Exception as e:
        return f"Erro ao processar a imagem: {e}"




def analisar_imagem_ollama(imagem_input, prompt="Descreva esta imagem em uma frase curta, foque em dar o contexto da situação para um sistema de segurança."):
    temp_path = None
    try:
        if isinstance(imagem_input, str):
            path_para_ollama = imagem_input
        elif isinstance(imagem_input, np.ndarray):
            rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = tmp.name
            img_pil.save(temp_path, format="PNG")
            tmp.close()
            path_para_ollama = temp_path
        else:
            raise ValueError("Tipo de entrada não suportado")
        response = chat(
            model="qwen2.5vl:3b",
            messages=[{"role": "user", "content": prompt, "images": [path_para_ollama]}],
        )
        return response.message.content
    except Exception as e:
        return f"Erro ao processar a imagem com Ollama: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def worker_analisar(frame_para_analisar):
    """
    Esta função será executada em uma thread separada.
    Ela chama analisar_imagem_gemini e imprime (ou armazena) o resultado.
    """
    
    print("chamada gemini")
    texto_descricao = analisar_imagem_gemini(frame_para_analisar)
    print(f"[Gemini] {texto_descricao}")
    
    #print("chamada ollama")
    #texto_descricao = analisar_imagem_ollama(frame_para_analisar)
    #print(f"[Qwen] {texto_descricao}")
    




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

cap = cv2.VideoCapture('b (1).mp4')
if not cap.isOpened():
    raise IOError("Não foi possível abrir o vídeo de entrada")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output_arthur.mp4', fourcc, fps, (width, height))
if not writer.isOpened():
    raise IOError("Não foi possível criar o vídeo de saída")




while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        classes=classes_desejadas,
        conf=CONF_THR,
        stream=False,
        verbose=False
    )

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
            ocorrencias +=1
            
            if ocorrencias == 2 and contexto_via_gemini == None:
                contexto_via_gemini = True
                
                frame_copy = frame.copy()  
                thread = threading.Thread(target=worker_analisar, args=(frame_copy,))
                thread.daemon = True
                thread.start()
            

    if results:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if time.time() < alert_end_time:
        cv2.putText(frame, f'State: {alert_class_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    writer.write(frame)
    cv2.imshow('Deteccao YOLO', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

