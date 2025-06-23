
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional

# Inicialização do modelo (fazer apenas uma vez)
model = YOLO('yolo11s.pt')
CONF_THR = 0.5
classes_desejadas = [0]  # [0, 1, 2, 3, 4]
#   "0": "person",
#   "1": "bicycle",
#   "2": "car",

def retirar_ruidos_frame(frame: np.ndarray) -> np.ndarray:
    """Remove ruídos do frame aplicando filtros."""
    frame = cv.medianBlur(frame, 3)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.equalizeHist(frame)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    return frame

def extrair_coordenadas_deteccao(results) -> List[Dict]:
    """Extrai coordenadas e informações das detecções."""
    deteccoes = []

    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            # Coordenadas da bounding box (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.astype(int)

            # Confiança
            conf = float(box.conf[0].cpu().numpy())

            # Classe
            cls = int(box.cls[0].cpu().numpy())

            # Calcular centro
            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)

            deteccao = {
                'id': i + 1,
                'classe': cls,
                'confianca': conf,
                'bbox': (x1, y1, x2, y2),
                'centro': (centro_x, centro_y),
                'largura': x2 - x1,
                'altura': y2 - y1
            }

            deteccoes.append(deteccao)

    return deteccoes

def detectar_objetos(frame: np.ndarray,
                    processar_frame: bool = True,
                    verbose: bool = False,
                    retornar_frame_anotado: bool = False) -> Dict:
    """
    Detecta objetos no frame usando YOLO.

    Args:
        frame: Frame de entrada (numpy array)
        processar_frame: Se deve aplicar pré-processamento no frame
        verbose: Se deve imprimir informações das detecções
        retornar_frame_anotado: Se deve retornar o frame com anotações

    Returns:
        Dict contendo:
        - 'deteccoes': Lista com informações das detecções
        - 'frame_anotado': Frame com bounding boxes (se solicitado)
        - 'total_deteccoes': Número total de detecções
    """
    try:
        # Pré-processamento opcional
        if processar_frame:
            frame_processado = retirar_ruidos_frame(frame)
        else:
            frame_processado = frame

        # Predição
        results = model.predict(
            source=frame_processado,
            classes=classes_desejadas,
            conf=CONF_THR,
            stream=False,
            verbose=False  # Sempre False para não travar
        )

        # Extrair coordenadas
        deteccoes = extrair_coordenadas_deteccao(results)

        # Print opcional das informações
        if verbose and deteccoes:
            print(f"=== {len(deteccoes)} DETECÇÃO(ÕES) ENCONTRADA(S) ===")
            for det in deteccoes:
                print(f"  Classe: {det['classe']}")
                print(f"  Confiança: {det['confianca']:.3f}")
                print(f"  Coordenadas: {det['bbox']}")
                print(f"  Centro: {det['centro']}")
                print("-" * 40)
        elif verbose:
            print("Nenhuma detecção encontrada!")


        resultado = {
            'deteccoes': deteccoes,
            'total_deteccoes': len(deteccoes),
            'sucesso': True
        }

        # Frame anotado opcional
        if retornar_frame_anotado:
            resultado['frame_anotado'] = results[0].plot()

        return resultado

    except Exception as e:
        print(f"Erro na detecção: {str(e)}")
        return {
            'deteccoes': [],
            'total_deteccoes': 0,
            'sucesso': False,
            'erro': str(e)
        }
