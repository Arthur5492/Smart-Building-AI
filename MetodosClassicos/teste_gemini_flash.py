import cv2
import numpy as np
import time


from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image


import threading


load_dotenv()


api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)


ocorrencias = 0
contexto_via_gemini = None

# prompt = "Descreva esta imagem de forma objetiva e concisa para um sistema de seguranca. Verifique se ha uma pessoa tentando entrar no portao, entrando no portao com bicicleta, ou apenas passando sem interagir. Seja direto, evite ambiguidades e use apenas letras sem acentos. "
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


def analisar_imagem_gemini(imagem_input, prompt=prompt):
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
