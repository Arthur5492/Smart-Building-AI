# Análise de Vídeo com IA

Este projeto utiliza detecção de objetos em tempo real (YOLO) para monitorar vídeos. Quando um objeto de interesse permanece na mesma região por um período, um frame é capturado e enviado para um modelo de linguagem multimodal (Google Gemini ou Qwen) para análise e descrição contextual da cena.

## Scripts

### `teste_gemini_flash.py`

Este script processa um arquivo de vídeo para detectar objetos pré-definidos. Se um objeto detectado permanece relativamente imóvel por um número específico de frames, o script aciona uma chamada para a API do Google Gemini. O frame é enviado para o modelo, que retorna uma descrição textual da cena, exibida no console. O vídeo resultante com as detecções é salvo como `output_arthur.mp4`.

### `teste_qwen_tela.py`

Similar ao script anterior, esta versão aprimorada também realiza a detecção de objetos, mas com funcionalidades adicionais. Quando uma análise é acionada, ele pode usar tanto a API do Gemini quanto um modelo local Qwen (via Ollama). A principal diferença é que a descrição gerada pela IA é sobreposta e exibida diretamente no vídeo de saída, com quebra de linha automática para textos longos e suporte a caracteres UTF-8 (necessita de um arquivo de fonte `.ttf`).

## Dependências

O projeto utiliza as seguintes bibliotecas Python:

* `opencv-python`
* `numpy`
* `ultralytics`
* `python-dotenv`
* `google-generativeai`
* `Pillow`
* `ollama`

## Instalação

Para instalar todas as dependências necessárias de uma só vez, execute o seguinte comando no seu terminal:

```bash
pip install opencv-python numpy ultralytics python-dotenv google-generativeai Pillow ollama
```

## Como Usar

1. **Crie um arquivo `.env`** na raiz do projeto para armazenar sua chave de API:
   ```
   GEMINI_API_KEY="SUA_CHAVE_API_AQUI"
   ```
2. **Coloque um arquivo de vídeo** na raiz do projeto (ex: `entregador.mp4`).
3. **Ajuste o nome do arquivo de vídeo** de entrada dentro do script que deseja executar.
4. **(Opcional para `teste_qwen_tela.py`)** Baixe uma fonte TrueType (como a [DejaVu Sans](https://dejavu-fonts.github.io/)) e coloque o arquivo `DejaVuSans.ttf` no mesmo diretório.
5. **Execute o script** via terminal:
   ```bash
   python teste_qwen_tela.py
   ```

### Usando o Ollama

Para usar um modelo localmente com Ollama (como o `qwen2.5vl:3b` utilizado no script `teste_qwen_tela.py`), siga estes passos:

1. **Instale o Ollama** no seu sistema a partir do [site oficial](https://ollama.com/).
2. Após a instalação, execute o seguinte comando no seu terminal para baixar e carregar o modelo. Este comando deixará o modelo pronto para uso em segundo plano:
   ```bash
   ollama run qwen2.5vl:3b
   ```
3. Uma vez que o modelo esteja rodando, o script Python conseguirá se conectar automaticamente ao serviço do Ollama que está ativo localmente para processar as imagens.
