## Instruções
 <ul>
    <li>Colocar dataset em data/datasets, certificar-se de que feature alvo é "classe"</li>
    <li> Por default, o tratamento para colunas categóricas é a aplicação de label encondig.</li>
 </ul> 

## TODO
<ul>
   <li>Existe a necessidade de gerar uma metabase nova para cada (modelo base) x (métrica de performance)? Muitos dados são repetidos</li>
   <li>Colocar kfold no nível base</li>

<ul>


## Feitos
<ul>
   <li>Simplifiquei bastante o data loader</li>
   <li>Passei o elbow como MFe</li>
</ul>


## Questionamentos 
<ul>
   <li>Há a necessidade do get_scaled na classe do Clustering Metrics?</li>
   <li>Implementei connectivity e  variações do size_dist(faz sentido?)</li>
</ul>