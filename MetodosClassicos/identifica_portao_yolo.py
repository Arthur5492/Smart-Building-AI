import cv2 as cv
import numpy as np
from teste_gemini_flash import analisar_imagem_gemini
from yolo_detectar_pessoa import detectar_objetos
from teste_ollama import analisar_imagem_ollama
import threading
from PIL import Image, ImageDraw, ImageFont

def main():
    video = cv.VideoCapture("./videos/portao_bike-ezgif.com-resize-video.mp4")

    if not video.isOpened():
        print("Error opening video file")
        return

    fps = video.get(cv.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    roi_x, roi_y, roi_w, roi_h = 820, 150, 150, 456  
    global gate_state
    global frame_count
    global qtd_frames_para_abrir_fechar
    global gemini_msg

    gemini_msg = None
    gate_state = "closed"  
    frame_count = 0
    movement_threshold = 1000  
    stable_frames_threshold = 30  
    qtd_frames_em_movimento = 0
    qtd_segundos_para_abrir_fechar = 10
    qtd_frames_para_abrir_fechar = qtd_segundos_para_abrir_fechar * fps
    gemini_detect = True
    gemini_delay_threshold = 100  


    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    
    qtd_deteccoes = 0

    def processa_frames(frame_count):
        ret, temp_frame = video.read()
        frame_count += 1
        current_time_seconds = frame_count / fps
        if not ret:
            return False, False, False
        
        temp_frame = retirar_ruidos_frame(temp_frame)

        cv.putText(temp_frame, f"State: {gate_state}, frame: {frame_count}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.rectangle(temp_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        if gemini_msg:
            temp_frame = put_text_with_background(temp_frame, gemini_msg, org=(10, height - 20), font_path=FONT_PATH, font_size=20, color=(255, 255, 255), max_width=width - 20)        
        writer.write(temp_frame)
        cv.imshow("Frame", temp_frame)
        cv.waitKey(30) 
        
        temp_roi = temp_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        temp_fgmask = bg_subtractor.apply(temp_roi)

        return temp_fgmask, current_time_seconds, frame_count

    def verificar_estado_portao_stable_frames_moviment(estado_possivel, frame_count):

        frames_min = 0
        # print(f"Verifying gate state: {estado_possivel}")
        # print(f"Stable frames threshold: {stable_frames_threshold}")
        for _ in range(stable_frames_threshold):
            temp_fgmask, current_time_seconds, frame_count = processa_frames(frame_count)
            if temp_fgmask is False:
                print("Error reading frame or end of video.")
                return frame_count, gate_state
            if cv.countNonZero(temp_fgmask) > movement_threshold:
                frames_min += 1
            if (frames_min >= stable_frames_threshold):
                print(f"Frame {frame_count}: Gate  {estado_possivel}")
                print(f"Time {current_time_seconds:.2f}s: Gate opened")  # Mostrar tempo em segundos
                return frame_count, estado_possivel
        
        return frame_count, gate_state
    
    def verificar_estado_portao_stable_frames_stopping(estado_possivel, frame_count, qtd_frames_em_movimento):

        frames_min = 0
        # print(f"Verifying gate state: {estado_possivel}")
        # print(f"Stable frames threshold: {stable_frames_threshold}")
        for _ in range(stable_frames_threshold):
            temp_fgmask, current_time_seconds, frame_count = processa_frames(frame_count)
            if temp_fgmask is False:
                print("Error reading frame or end of video.")
                return False, False, False
            if cv.countNonZero(temp_fgmask) < movement_threshold//2:
                frames_min += 1
        if (frames_min >= stable_frames_threshold):
            # print("qtd_frames_em_movimento:", qtd_frames_em_movimento)
            # print("qtd_frames_para_abrir_fechar:", qtd_frames_para_abrir_fechar)
            print(qtd_frames_em_movimento < qtd_frames_para_abrir_fechar)
            if (int(qtd_frames_em_movimento) < int(qtd_frames_para_abrir_fechar)):
                print(f"Frame {frame_count}: Gate  left_open")
                print(f"Time {current_time_seconds:.2f}s: Gate left_open")  # Mostrar tempo em segundos
                return frame_count, "left_open", qtd_frames_em_movimento
            else: 
                print(f"Frame {frame_count}: Gate  {estado_possivel}")
                print(f"Time {current_time_seconds:.2f}s: Gate {estado_possivel}")  # Mostrar tempo em segundos
                return frame_count, estado_possivel, 0

            

        return frame_count, gate_state, qtd_frames_em_movimento
    
    def worker_analizar(roi, tipo="gemini"):
        global gemini_msg
        if tipo == "ollama":
            response = analisar_imagem_ollama(roi)
            print(f"[Ollama] {response}")
            gemini_msg = response
        else:
            response = analisar_imagem_gemini(roi)
            print(f"[Gemini] {response}")
            gemini_msg = response

    FONT_PATH = "DejaVuSans.ttf" 

    def put_text_with_background(frame, text, org, font_path, font_size, color, max_width):
            try:
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            except IOError:
                cv.rectangle(frame, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
                error_msg = "ERRO: Fonte .ttf nao encontrada. Caracteres especiais nao serao exibidos."
                cv.putText(frame, error_msg, (10, frame.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv.putText(frame, text, (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return frame

            frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
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

            return cv.cvtColor(np.array(frame_pil), cv.COLOR_RGB2BGR)

    def retirar_ruidos_frame(frame):

        color = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 3)
        gray = cv.equalizeHist(gray)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        frame = cv.addWeighted(color, 0.5, gray, 0.5, 0)

        return frame

    nome_video_saida = "saida_portao.mp4"
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(nome_video_saida, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Error opening video writer for {nome_video_saida}")
        return

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        frame_count += 1
        current_time_seconds = frame_count / fps  # Converter frame para segundos

        if  frame_count % gemini_delay_threshold == 0:
            gemini_detect = True

        frame = retirar_ruidos_frame(frame)

        cv.putText(frame, f"State: {gate_state}, frame: {frame_count}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        
        if gemini_msg:
            frame = put_text_with_background(frame, gemini_msg, org=(10, height - 20), font_path=FONT_PATH, font_size=20, color=(255, 255, 255), max_width=width - 20)        
        writer.write(frame)

        cv.imshow("Frame", frame)
        
        if cv.waitKey(30) == ord('q'):
            break

        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        fgmask = bg_subtractor.apply(roi)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        
        movement = cv.countNonZero(fgmask)
        
        cv.imshow("Foreground Mask", fgmask)
        

        if gate_state == "open":
            qtd_frames_em_movimento += 1

        if movement > movement_threshold:

            detect_pessoa = detectar_objetos(roi, processar_frame=False, verbose=False, retornar_frame_anotado=True)

            
            if detect_pessoa['total_deteccoes'] > 0:
                # print(detect_pessoa)
                # print(f"Frame {frame_count}: Detected {detect_pessoa['total_deteccoes']} person(s)")
                    
                if gemini_detect:
                    print(f"Frame {frame_count}: Person detected")
                    
                    # pegar frame anterior para analisar com o gemini
                    if qtd_deteccoes < 5:
                        thread = threading.Thread(target=worker_analizar, args=(roi,))
                        thread.daemon = True
                        thread.start()
                        qtd_deteccoes +=1
                    gemini_detect = False


            if gate_state == "left_open":
                frame_count, gate_state = verificar_estado_portao_stable_frames_moviment("open", frame_count)

            if gate_state == "closed":

                frame_count, gate_state = verificar_estado_portao_stable_frames_moviment("open", frame_count)

        #checar se o portão foi deixado aberto
        elif movement < movement_threshold//2:  
            
            if gate_state == "open":
                
                frame_count, gate_state, qtd_frames_em_movimento = verificar_estado_portao_stable_frames_stopping("closed", frame_count, qtd_frames_em_movimento)
       
    
    #salvar video

    
    writer.release()

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()