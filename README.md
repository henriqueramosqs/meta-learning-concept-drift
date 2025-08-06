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
<li> Tem avaliação da off e da on -> Cuidar para salvar as infos necessárias para as avaliações</li>
<li> Analisar se consigo resultados similares que os da Fernanda na parte off</li>
__
<li>Terminei a parte offline</li>
<li>Terminei a classe de base_data_manager</li>
<li>Terminei a classe de meta_data_manager</li>
<li>https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#effect-of-rescaling-on-a-pca-dimensional-reduction</li>
<li>Faz sentido reduzir dimensões do df inteiro ao invés de só o batch?</li>
<li>learning_window_size faz sentido? Era 94 no caso que tá no  jupyter -> ETA?</li>
<li>Pq eu espero acumular STEP novos na metabse? (setp*step instâncias no base)</li>
<li>Kurtosis e skewness não retornam NaN</li>
<li>Terminei o experimento</li>
<li>Comparei os resultados com o que a Fernanda fez</li>

