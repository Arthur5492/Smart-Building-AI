# Smart-Building-AI

Repositório de notebooks Jupyter e scripts Python para aplicação de Inteligência Artificial em cenários de edifícios inteligentes.

Arthur Rodrigues Fernandes - arthur.r.f@grad.ufsc.br

Sandy Hoffmann - sandy.hoffmann@posgrad.ufsc.br

Vinícius Wolosky Muchulski - vinicius.muchulski@grad.ufsc.br


## Visão Geral

O **Smart-Building-AI** explora como técnicas de IA e análise de dados podem ser utilizadas para otimizar e automatizar aspectos de segurança e operação de edifícios inteligentes. Os notebooks e scripts incluem exemplos de:

- Análise e processamento de imagens de vídeo
- Detecção de pessoas e objetos usando modelos YOLO
- Geração de contexto e análise de situações via modelos generativos (Google Gemini, Ollama)
- Automatização de tarefas de segurança

## Instalação dos Pacotes Necessários

Os principais pacotes Python a serem instalados para executar os notebooks e scripts estão listados abaixo:

```bash
pip install opencv-python numpy ultralytics python-dotenv google-generativeai pillow ollama
```

> **Observações:**
> - Os módulos `threading`, `os`, `tempfile` e `time` já fazem parte da biblioteca padrão do Python.
> - Baixe o arquivo de fonte `"DejaVuSans.ttf"` e coloque no mesmo diretório dos scripts que o utilizam.
> - Os modelos YOLO utilizados (ex: `yolo11n.pt`, `yolo11s.pt`) devem estar presentes no diretório ou serem baixados conforme documentação do YOLO/Ultralytics.
> - Para uso das APIs do Gemini, crie um arquivo `.env` com sua chave de API:  
>   ```
>   GEMINI_API_KEY=SEU_TOKEN_AQUI
>   ```
> - O pacote Ollama pode exigir instalação e configuração adicional; verifique a [documentação oficial](https://ollama.com/).

## Como Usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Arthur5492/Smart-Building-AI.git
   cd Smart-Building-AI
   ```

2. (Opcional) Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale os pacotes necessários (veja acima).

4. Execute os notebooks com:
   ```bash
   jupyter notebook
   ```
   Ou rode scripts Python diretamente.

## Contribuição

Contribuições são bem-vindas! Abra issues ou pull requests para novas funcionalidades, correções ou melhorias.

## Licença

Este projeto ainda não possui uma licença definida. Para uso ou contribuições, entre em contato com o proprietário do repositório.

---

> **Dica:** Navegue pelos notebooks Jupyter e scripts para ver exemplos de uso e funcionalidades implementadas.
