import cv2
import numpy as np
import os
from ollama import chat
import tempfile
from PIL import Image

prompt = """Analise esta imagem para um sistema de seguranca. Siga estas regras estritamente:
1. Descreva apenas pessoas proximas ao portao (ate 5 metros).
2. Identifique UMA destas acoes:
   - 'PESSOA_TENTANDO_ENTRAR' (se tentar abrir o portao)
   - 'PESSOA_ENTRANDO (se estiver entrando no portao)'
    - 'PESSOA_ENTRANDO_COM_BICICLETA (se estiver entrando no portao com bicicleta)'
   - 'PESSOA_PASSANDO_SEM_INTERAGIR'
   - 'NENHUMA_PESSOA_PROXIMA' (se nao houver pessoas relevantes)
3. Adicione apenas 1 detalhe fisico breve (ex: 'CAMISA_VERMELHA', 'CALCA_JEANS').
4. NUNCA use acentos ou caracteres especiais.
5. Formato obrigatorio: '[ACAO] + [DETALHE]' (ex: 'PESSOA_PASSANDO_SEM_INTERAGIR CAMISA_AZUL').

Resposta:"""


def analisar_imagem_ollama(imagem_input, prompt=prompt):
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
