# Approximation Algorithms

Cool approximation algorithms, along with their mathematical proofs are added here. 

Firstly, we cover some of the approximation algorithms for the metric and asymmetric traveling salesman problem. The metric TSP admits two classical results, namely, the double tree algorithm and Christofides algorithm. The asymmetric case includes the simple TSP heuristics (such as, cheapest insertion, nearest neighbor and the greedy algortihms) as we know the worst-case guarantees for these algorithms. Frieze et al.[^1] gave the first nontrivial $\log_{2} n$-approximation algorithm in 1982 which was subsequently improved to slightly better factors of $\log_{2} n$. This algorithm is referred to as the *Repeated Assignment Heuristic* or more recently (and commonly) as the *Cycle Cover Algorithm* or *Cycle Shrinking Algorithm*. The major breakthrough in this field was the $O(\log n / \log \log n)$ -approximation algorithm due to Asadpour et al. [^2] (breaking the 30 year barrier of improvements of the $O(log n)$-approximation). Recent approximation algorithms have found constant factor worst-case guarantees for the asymmetric case, though, they weren't covered in this work.

This work part of the project studies at the Chair of Operations Research, Technical University of Munich. Overall, we cover the works of Frieze et al. and Asadpour et al. in detail. The real world TSP instances were from [Amazon](https://registry.opendata.aws/amazon-last-mile-challenges/) as part of their Last Mile Delivery Routing challenge from 2018.

[^1]: Frieze, A., M., Galbiati, G., Maffioli, F. "On the Worst-Case Performance of Some Algorithms for the Asymmetric Traveling Salesman Problem" (1982)
[^2]: Asadpour, A., Goemans, M. X., Madry, A., Gharan, S., O., Saberia, A. "An O(log n/log log n)-approximation Algorithm for the Asymmetric Traveling Salesman Problem." (2017).
