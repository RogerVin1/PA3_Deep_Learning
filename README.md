# PA3_Deep_Learning

## 1. Baixar os dados

Os dados (~415 MB) não estão versionados no repositório. Baixe o arquivo `pa3_dados.zip` pelo Google Drive:

**https://drive.google.com/file/d/1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui/view?usp=sharing**

Descompacte o zip **na raiz do repositório**, de modo que a estrutura fique assim:

```
PA3_Deep_Learning/
├── PA3_DL.ipynb
├── requirements.txt
├── pa3/
├── scripts/
├── weights/
└── pa3_dados/
    ├── classes.csv
    ├── images/
    └── labels/
```

O código procura o arquivo `classes.csv` dentro da pasta informada. O notebook usa `DATA = "pa3_dados"` por padrão. Se os dados estiverem em outro lugar, altere essa variável.

Pelo terminal (opcional), usando o [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown
gdown 1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui -O pa3_dados.zip
unzip pa3_dados.zip -d pa3_dados
```

## 2. Instalar as dependências

Recomenda-se usar um ambiente virtual:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

No Google Colab, `torch` e `torchvision` já vêm instalados; basta rodar `!pip install -r requirements.txt` (ou instalar só os demais pacotes).

## 3. Executar

Abra o notebook `PA3_DL.ipynb` e execute as células em ordem. Os scripts em `scripts/` também podem ser rodados diretamente, por exemplo:

```bash
python scripts/part0_floor.py --data pa3_dados --budget 5 --seed 0
```
