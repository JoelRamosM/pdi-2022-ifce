Dado um conjunto contendo imagens de documentos capturadas por diversas cameras de smartphones em diferentes resoluções,
as quais devem ser preparadas utilizando de técnicas da área de processamento de imagens e sinais para reduzir a quantidade de erros ocorridos
na extração de textos utilizando técnicas de OCR e facilitar ao máximo a identificação dos textos de forma manual.

A presença de sombras no documento fotografado é uma das principais dificuldades para que se possa conferir manualmente,
visto que para mitigar o prolemas o custo com armazenamento fica relativamente alto, pois as imagens devem ser armazenadas em
resolução e dimensões originais normalmente superiores a 2268 X 3024 e 72dpi. Álém de melhorar a visualização para conferência manual,
os experimentos visam ressaltar as bordas do documento, pois pontos de muita luz, sombra pou sobre uma superficie branca
acabam escondendo as bordas do documento que em sua maioria são papéis brancos.

Os seguintes passos foram aplicados para tratamento da imagem:

+ Filtro passa-baixa, para suavizar e remover ruidos;
    + janela 5X5.
+ Binarização da imagem para remover sombras e diminuir efeito de feixos de luz;
    + definir limiar;
+ Dilatação -  para reforçar os limites das bordas do documento e remover falhas de caracteres impressos/digitados e/ou escritos a mão;
    + kernel de tamanho 3;



