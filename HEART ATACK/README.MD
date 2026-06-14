Em relação a escolha do modelo como era pra gente usar o meta estimador que quisessemos
eu me senti bem tentado ao usar o meu canivete suiço que eu tenho gostado bastante de usar
q é o xgboost, sim eu sei que você não gosta mas como a gente tem lidado muito com dados
numericos ele acaba sendo muito bom, mas nesse caso eu não me senti confotavel com ele

Eu considerei que primeiramente o dataset era bem pequeno, então com ctz absoluta
iria overfittar e MUUUUUUUITO ja q ele é otimizado pra big data e ia só decorar

alem de que pra esse caso lidar com ele não seria exatamente eficiente e ia fazer
eu ficar 2 horas tunando ele pra pouco resultado

# ENTAO PQ RANDOM FOREST?
Bem simplesmente pois a random forest pega todos esses problemas e janta eles com
farofa, as poucas amostras nn sãu um BO ja que rf generaliza bem, alem de lidar
com os outliers, e eu nem precisaria normalizar mas achei interessante pra dar
um esteroide de leve na performance

Alem de simplificar bastante a aplicação do modelo ja que a maioria deles
inevitavelmente ia me fazer perder muito mais tempo escovando os dados, portanto
achei que random forest seria uma escolha bem segura pra um dataset simples desse