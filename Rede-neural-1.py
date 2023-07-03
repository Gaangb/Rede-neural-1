#Este codigo serve para ler cartas/bilhetes escritas a mão

import numpy
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


transform = transforms.ToTensor() #definindo a conversão de imagem para tensor
trainset = datasets.MNIST('./MNINST.data', download=True, transform=transform) #Carrega a parte de treino do dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) #Cria um buffer para pegar os dados por partes

valset = datasets.MNIST('./MNIST_data', download=True, train=False, transform=transform) #Carrega a parte de validação
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True) #Cria um buffer para pegar os dados

dataiter = iter(trainloader) # Converte o trainloader em um iterador
imagens, etiquetas = dataiter.__next__() # Obtém o próximo batch de dados e suas respectivas etiquetas
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r'); # Exibe a imagem do primeiro exemplo do batch

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        
        # Definição das camadas da rede
        
        self.linear1 = nn.Linear(28*28, 128) # camada de entrada, 784 neuronios que se ligam a 128
        self.linear2 = nn.Linear(128, 64) # camada interna 1, 128 neuronios que se ligam a 64
        self.linear3 = nn.Linear(64,10) # camada interna 2, 64 neuronios que se ligam a 10
        # para a camada de saida não é necessario definir nada pois so precisamos pegar o output da camada interna 2

    
    def forward(self, x):
        # Implementação do fluxo de dados na rede
        
        x = F.relu(self.linear1(x)) # Função de ativação da camada de entrada para a camada interna 1
        x = F.relu(self.linear2(x)) # Função de ativação da camada interna 1 para a camada interna 2
        x = self.linear3(x) # Funcao de ativação da camada interna 2 para a camada de saida, nesse caso f(x) = x
        return F.log_softmax(x, dim=1)
    
def treino(modelo, trainloader, device):

    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # Define a politica de atualização dos pesos e da bias
    inicio = time() #timer para sabermos quanto tempo levou o treino

    criterio = nn.NLLLoss() # Definindo o criterio para calcular a perda
    EPOCHS = 10 # numero de epochs que o algoritmo rodará
    modelo.train() # ativando o modo de treinamento do modelo

    for epoch in range(EPOCHS):
        perda_acumulada = 0 # inicialização da perda acumulada da epoch em questao

        for imagens, etiquetas in trainloader:

            imagens = imagens.view(imagens.shape[0], -1)
            otimizador.zero_grad() #zerando os gradientes por conta do cilco anterior

            output = modelo(imagens.to(device)) #colocando os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) #calculando a perda da epoch em questao

            perda_instantanea.backward() # back progation a partir da perda

            otimizador.step() #atualizando os pesos e as bias

            perda_acumulada += perda_instantanea.item() #atualizacao da perda acumulada

        else:
            print("Epoch {} - Perda resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))

    print("/nTempo de treino (em minutos) s", (time()-inicio)/60)



def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0
    for imagens, etiquetas in valloader:
        for i in range(len(etiquetas)):
            img = imagens[i].view(1, 784)
            #desativar o autograd para acelerar a validacao. Grafos computacionais dinaicos tem um custo alto de processamento
            with torch.no_grad():
                logps = modelo(img.to(device)) # output do modelo em escala logaritmica
            

            ps = torch.exp(logps) # converte output para escala normal(lembrando que é um tensor)
            probab = list(ps.cpu().numpy()[0])
            etiqueta_pred = probab.index(max(probab)) # converte o tensor em um numero, no caso, o numero que o modelo previu
            etiqueta_certa = etiquetas.numpy()[i]
            if(etiqueta_certa == etiqueta_pred): # compra a previsao com o valor correto
                conta_corretas +=1
            conta_todas += 1
        
        print("Total de imagens testadas =", conta_todas)
        print("\nPrecisão do modelo = {}%".format(conta_corretas*100/conta_todas))



modelo = Modelo() #inicializa o modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #modelo que rodara na GPU se possivel
modelo.to(device)
treino(modelo, trainloader, device)  # Chamando a função de treinamento
validacao(modelo, valloader, device)  # Chamando a função de validação