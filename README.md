## Instruções
 <ul>
    <li>Colocar dataset em data/datasets, certificar-se de que feature alvo é "classe"</li>
    <li> Por default, o tratamento para colunas categóricas é a aplicação de label encondig.</li>
 </ul> 

## Feitos
<ul>
   <li>Simplifiquei bastante o data loader</li>
   <li>Passei o elbow como MFe</li>
   <li>Incorprei novas métricas de cluster (connectivity e sizeDist)</li>
   <li>Implementei variancia</li>
   <li>Implementei o nrCorAttr -> perguntar por que multiplicar por 2</li>
   <li>Sparsity vs attributes sparsity</li>
</ul>


## Questionamentos 
<ul>
   <li>Eu não deveria fazer a transformação get_scaled em toda métrica ao invés de apenas no Clustering Metrics?</li>
   <li>Implementei connectivity e  variações do size_dist(faz sentido?)</li>
   <li>Por que só tinha compactness no kmeans?</li>
   <li>Posso rodar um Meta Learning para cada combinação (dataset)x(include_new_mfes) ao invés de (dataset) x (performance_metric) x(base_model) x (include_performance_metric)
   </li>
   <li>Algum motivo para kurtosis e skewness não serem utilizados? (exemplo da acurácia)</li>
   <li>Covariância é descartável?</li>
   <li>prop_pca estava certo?</li>
   __
   <li>Pensar bem sobre a forma de avaliar o execution time -> problema do cold heat, etc</li>   
   <li>Há a necessidade do copy?</li>   
</ul>



## Notas

<li> Normalização de dados pro DBSCAN</li>
<li> Fiz Results_importace e result_gain</li>
<li> Results_gain "no olho" parecido com o da Fernanda -> Como comparar? Subtrair um do outro? Considerar o dela outro conjunto?</li>
<li> Feature importances </li>
   <li> DBSCAN fez uma contribuição significativa</li>
   <li> Das que eu coloquei, ADWIN teve mais protagonismo</li>
   <li> Senti falta do KSWIN -> Nos experimentos anteriores que eu fiz, lembro que ele teve mais importância. Também era o esperado para ter mais significância</li>
   <li> Outros soterrados</li>
<li> Como foi escolhido o feature fraction? </li>
<li> Perguntar sobre a necessidade de outro treinamento para feature importance  </li>
<li> Matérias Pós </li>
<li> SLM </li>
<li> Perguntar sobre TCC </li>
 
<h3></h3>
 <li> MSE != ganho. Baseline = mais recomendado das janelas anteriores. Prestar bem atenção no código da Fernanda</li>
<li> Colocar gráficos lado a lado -> Ver consistência</li>
<li> Fazer experimentos sem "fortalecer" as métricas originais(N)</li>
<li> Fortalecer as originais e analisar o feature importance separando-as(N)</li>
<li> Parte II</li>

No conjunto de métricas originais, eu fiz as seguintes alterações:

<h3>Stats Metrics</h3>
<li>sparsity (diferente de attributes sparsity!)</li>
<li>Variância</li>
<li>kurtosis e skewness</li>
<li>NrAtributos Correlatos</li>
<h3>Clustering Metrics</h3>
<li>DBSCAN</li>
<li> Elbow do kmeans</li>
<li> min_size_dist</li>
<li> max_size dist</li>
<li> mean_size_dist</li>
<li> connectivity</li>

Não voltei pra original: Normalizações e correção do PCA

* Todas as métricas com excessão do kappa estavam iguais