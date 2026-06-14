Ok vamo la que esse aqui me estressou, quando eu bati o olho em 100 mil linhas eu ja pensei que ia ter dado 2 horas rodando e nada
ia ter sido treinado, o que me levou pra minha obsecção em ml ultimamente de velocidade, e eu quis MUITO usar o histboost aqui
mas teve um grande porem que me fez repensar, que o xgboost é otimizado pra big data enquanto o hist simplesmente é mais ou menos
mas ele tem um monte de problema como a falta de cross validation nativa do hist e que com certeza ele iria overfittar pelo menos um pouco

levando isso em conta, os dois dariam uma boa briga mas o xgboost muito provavelmente iria melhor, e como eu ja teria que separar em varios
treinos pelo tamanho, eu achei melhor nem inventar moda.

alem do e uma coisa que eu não comentei ainda nos outros readmes (se eu não me engano) que é o fato do xgb llidar bem com  muitos categoricos
(que eu apelidei na minha cabeçad de 3d) ele manda muito bem nisso e praticamente nasceu pra isso, então ele capturaria muito bem os dados
então me rendi mais uma vez a ele e achei que ele mandou muito bem pra um dataset bem titanico.