Pois bem, aqui eu quis experimentar fazer algo um pouco diferente e testar um metaestimador que eu nunca tinha usado antes
mas que vi pesquisando pra me preparar pra segunda parte da A2, que é o hist gradient boosting (sim eu amo gradient boosting)
ele é pra ser o mais mais rapido mega turbo que tem, o que me agradou bastante e queria avaliar a performance dele antes de
partir pro blackfriday ja que ele seria IMENSO.

Ja o o random forest tem as facilidades de sempre de ser no minimo ok pra todos os datasets ent eu gosto dele como uma baseline
e o xgboost seria o competidor direto dele e os dois fazem o par do meu maior interesse.

Mas na hora de finalmente treinar o resultado foi meio mistozinho, todos foram no minimo ok entre os 3 mas o hist
realmente não foi tão mais rapido quanto eu esperava, ainda sendo em menos de 1 segundo pro final, o que me deixou bem impressionado
mas ao mesmo tempo não senti tanta potencia assim, não o suficiente pra fazer valer a pena em datasets menores como esse, e eu to meio acostumado a usar o
xgboost a esse ponto

Melhores parametros HGB: {'min_samples_leaf': 10, 'max_leaf_nodes': None, 'max_iter': 500, 'max_depth': 10, 'learning_rate': 0.2, 'l2_regularization': 0.5}
Melhor score CV HGB (f1_weighted): 0.8856

Tempo de treino XGB: 2.9479 segundos

Tempo de treino RF: 1.6951 segundos

pçra não falar que eu to mentindo

e querendo ou não quem ganhou na acuracia absoluta foi o xgboost, não por muito mais ainda foi o MELHOR em numeros absolutos, ainda que por uma margem menor
então apesar de eu até ter gostado do hgb eu fui de xgb mesmo, apesar de ter ficado impressionado a unica justificativa tecnica que importa aqui seriam os dados
brutos, e nisso o xgb ta um passinho atras, lembrando que o meu cuidado com o tuning dos 3 foi igual.