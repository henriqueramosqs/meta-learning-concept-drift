## Instruções
 <ul>
    <li>Colocar dataset em data/datasets, certificar-se de que feature alvo é "classe"</li>
    <li> Por default, o tratamento para colunas categóricas é a aplicação de label encondig.</li>
 </ul> 

## TODO
<ul>
   <li>Existe a necessidade de gerar uma metabase nova para cada (modelo base) x (métrica de performance)? Muitos dados são repetidos</li>
   <li>Colocar kfold no nível base</li>

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
   <li>
   
</ul>