
clear; 
close(windisd());
clc; 

// Defino el directorio
cd "C:\Users\gabi\Documents\FACULTAD\Maestria Metodologia\modelos de simulacion\datos"

// datos mensuales desde enero94 hasta diciembre 24 de: precio_soja_dolares ; precio petroleo ; tasa de interes bonos US ; indice de tcr USA ; indice SP500, indice DJI ; IPC ; Exportaciones mundiales
datos = csvRead('df111.csv');
datos = datos(2:$,2:$)


// Normalizar los datos
mu = mean(datos,"r");          // vector fila de medias
sigma = stdev(datos,"r");      // vector fila de desvíos
datos_n = (datos - ones(size(datos,1),1)*mu) ./ (ones(size(datos,1),1)*sigma);

// gráficos
close(winsid())
scf()
a.blackground=-2
subplot(331)
plot(datos_n(:,1))
title('Precio soja')
subplot(332)
plot(datos_n(:,2))
title('Precio petroleo')
subplot(333)
plot(datos_n(:,3))
title('Tasa de interés')
subplot(334)
plot(datos_n(:,4))
title('Tipo de cambio')
subplot(335)
plot(datos_n(:,5))
title('S&P500')
subplot(336)
plot(datos_n(:,6))
title('Down Jones')
subplot(337)
plot(datos_n(:,7))
title('IPC')
subplot(338)
plot(datos_n(:,8))
title('X mundial')


// Primeras diferencias
D1 = zeros(size(datos_n,1), size(datos_n,2));

for k = 1:size(datos_n, 2)
    for i = 2:size(datos_n,1)
    D1(i,k) = datos_n(i,k)-datos_n(i-1,k);
    end
end

precio = datos_n(:,1);
train = datos_n(:,2:size(datos_n,2));
train = [train D1];

// Agrego rezagos
lags = 2; // números de rezagos
N = size(train, 1);

train_DL = []
for k=1:lags
    train_DL = [train_DL train(lags+1-k:N-k,1:size(train,2))]
end

precio_DL = precio(3:$)

// Determinantes del modelo
N = [size(train_DL, 2) 20 1];
lr = 0.05;
factivIH = "ann_purelin_activ";
factivHH = "ann_purelin_activ";
factivHO = "ann_purelin_activ";
epocas = 10000

// Entreno modelo
scf(2);
W = ann_FFBP_gda(train_DL', precio_DL', N, [factivIH, factivHO], lr, [], [], epocas, [], [], []);


// estimo la serie que predice
n_test = size(train_DL, 1);
pred = zeros(n_test,1);

for i = 1:n_test
    pred(i) = ann_FFBP_run(train_DL(i,:)', W, [factivIH, factivHO]);
end

//Grafico
close(windisd());
plot(pred, '-r')
plot(precio_DL, 'b-')


