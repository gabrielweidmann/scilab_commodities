// Sintaxis para entrenar ANN con las variables mencionadas. Se define parametro "t" que significa la distancia entre el mes observado y el precio explicado, y "lags" la cantidad de rezagos (en meses inmediatos consecutivos para atras). Luego entrena y muestra la ANN

clear; 
close(winsid());
clc; 

// Defino el directorio
cd "C:\Users\gabi\Documents\FACULTAD\Maestria Metodologia\modelos de simulacion\datos"

// datos mensuales
//listado de variables: precio soja en t+1; precio soja; precio petroleo, tasa de interes ; tipo de cambio real ; SP500 ; Down Jones ; IPC ; precio fertilizantes ; consumo de soja mundial ; stock de soja mundial ; precio de soja mundial ; exportaciones de soja mundial

datos = csvRead('dfmas1.csv');
datos = datos(2:$,4:$);


// Definiciones
t = 3 //Distancia hacia adelante de precio a predecir
lags = 2 // número de lags incorporados 

// Normalizar los datos
mu = mean(datos,"r");          // vector fila de medias
sigma = stdev(datos,"r");      // vector fila de desvíos
datos_n = (datos - ones(size(datos,1),1)*mu) ./ (ones(size(datos,1),1)*sigma);

// Adecuo según el valor t definido
precio = datos_n((1+t):$,1); // vector precio a explipcar
train = datos_n(1:$-t,2:size(datos_n,2)); // data frame recortando los últimos t valores, que se quedan sin precio asignado


// Primeras diferencias
D1 = diff(train,1,"r");
train = [train(2:$,:) D1];


// Agrego rezagos
N = size(train, 1);

train_DL = []
if lags == 0 then
    train_DL = train;
    else
        for k=1:lags
            train_DL = [train_DL train(lags+1-k:N-k,1:size(train,2))];
        end
end

precio_DL = precio((2+lags):$);

// Determinantes del modelo
N = [size(train_DL, 2) 20 1];
lr = 0.05;
factivIH = "ann_purelin_activ";
factivHH = "ann_purelin_activ";
factivHO = "ann_purelin_activ";
epocas = 5000

// Entreno modelo
scf(2);
W = ann_FFBP_gda(train_DL', precio_DL', N, [factivIH, factivHO], lr, [], [], epocas, [], [], []);


// estimo la serie que predice incorporando nueva bd completa, para proyectar hacia meses sin precio observado

proy = datos_n(:,2:size(datos_n,2)); // data frame recortando los últimos t valores, que se quedan sin precio asignado


// Primeras diferencias
D1_p = diff(proy,1,"r");
proy = [proy(2:$,:) D1_p];


// Agrego rezagos
N = size(proy, 1);

proy_DL = []
if lags == 0 then
    proy_DL = proy;
    else
        for k=1:lags
            proy_DL = [proy_DL proy(lags+1-k:N-k,1:size(proy,2))];
        end
end




n_test = size(proy_DL, 1);
pred = zeros(n_test,1);

for i = 1:n_test
    pred(i) = ann_FFBP_run(proy_DL(i,:)', W, [factivIH, factivHO]);
end

//Grafico
scf(3);
plot(pred, 'r-')
plot(precio_DL, 'b-')
legend(["Proyectado", "Observado"], 3);



